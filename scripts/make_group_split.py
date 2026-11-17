"""
make_group_split.py - Leakage-free train/val/test split for image datasets
that contain duplicates and near-duplicates (e.g. frames from the same video
or the same patient).

How it works:
  1. Reads every image under the dataset folder (class = parent folder name).
     Any existing split folders (Training / Testing ...) are ignored; all images
     are pooled together.
  2. Exact duplicates (same bytes) are removed - one copy is kept.
     If copies have DIFFERENT labels they are all dropped and reported.
  3. Near-duplicates are found with a perceptual hash (dHash) computed on the
     CENTER of the image (otoscope images have a black circular border that
     makes whole-image hashes look similar).
  4. Near-duplicates are linked into groups (connected components).
     A group is treated like "one patient": the whole group goes to ONE split.
  5. Groups are assigned to train/val/test so each class gets roughly the
     requested ratio (stratified group split).
  6. Verifies that no near-duplicate pair crosses splits.

Outputs (in --out folder):
  split.csv              path, class, group, split  (use this in your training code)
  split_summary.txt      counts per split/class, group sizes, removed files
  cross_class_pairs.png  near-duplicates with different labels (check by eye)
  (optional) --copy      copies images into out/train|val|test/<class>/

Usage (Windows):
  python scripts\\make_group_split.py data\\ear\\chile\\Datos --out data\\ear\\chile\\split_v1
  python scripts\\make_group_split.py data\\ear\\chile\\Datos --out data\\ear\\chile\\split_v1 --copy
"""

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

try:
    import numpy as np
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Missing packages. Run:  pip install pillow numpy matplotlib")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def center_dhash(img, crop=0.6, size=8):
    """dHash of the central crop -> ignores the black otoscope border."""
    w, h = img.size
    cw, ch = int(w * crop), int(h * crop)
    left, top = (w - cw) // 2, (h - ch) // 2
    g = img.crop((left, top, left + cw, top + ch)).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def load_images(root):
    items = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            cls = os.path.basename(dirpath)
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    items.append({"path": rel, "class": cls, "md5": md5_of(path),
                                  "hash": center_dhash(im)})
            except Exception as e:  # noqa: BLE001
                print(f"  skip unreadable: {rel} ({e})")
    return items


def assign_splits(groups, ratios, seed):
    """Greedy stratified group assignment.
    groups: list of lists of item dicts. Returns {group_index: split_name}."""
    names = list(ratios)
    class_total = Counter(it["class"] for g in groups for it in g)
    target = {s: {c: class_total[c] * ratios[s] for c in class_total} for s in names}
    current = {s: Counter() for s in names}

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: -len(groups[i]))  # big groups first, ties random

    result = {}
    for gi in order:
        comp = Counter(it["class"] for it in groups[gi])
        best, best_score = None, None
        for s in names:
            # how far below target this split would still be for these classes (higher = more room)
            score = sum((target[s][c] - current[s][c]) / max(class_total[c], 1) * n for c, n in comp.items())
            if best_score is None or score > best_score:
                best, best_score = s, score
        result[gi] = best
        current[best].update(comp)
    return result


def save_cross_class_sheet(root, pairs, out_path, limit=24):
    if not pairs:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pairs = pairs[:limit]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)), squeeze=False)
    for i, (a, b, d) in enumerate(pairs):
        for j, it in enumerate((a, b)):
            ax = axes[i][j]
            ax.axis("off")
            with Image.open(os.path.join(root, it["path"])) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{it['class']}\n{os.path.basename(it['path'])}  (d={d})", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Leakage-free stratified group split.")
    ap.add_argument("root", help="Dataset folder (class = parent folder of each image)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--threshold", type=int, default=6,
                    help="Max dHash distance to treat two images as near-duplicates (default 6)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Also copy images into out/<split>/<class>/")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    total_ratio = args.train + args.val + args.test
    ratios = {"train": args.train / total_ratio, "val": args.val / total_ratio, "test": args.test / total_ratio}
    ratios = {k: v for k, v in ratios.items() if v > 0}
    os.makedirs(args.out, exist_ok=True)

    print("Reading images...")
    items = load_images(root)
    print(f"  {len(items)} images, classes: {dict(Counter(i['class'] for i in items))}")

    # ---- 1. exact duplicates
    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it)
    kept, removed_dup, removed_conflict = [], [], []
    for group in by_md5.values():
        if len({g["class"] for g in group}) > 1:
            removed_conflict.extend(group)       # same file, different labels -> unreliable
            continue
        kept.append(group[0])
        removed_dup.extend(group[1:])
    print(f"  removed {len(removed_dup)} exact duplicates, {len(removed_conflict)} conflicting-label copies")

    # ---- 2. near-duplicate groups
    n = len(kept)
    uf = UnionFind(n)
    near_pairs = []
    for a in range(n):
        ha = kept[a]["hash"]
        for b in range(a + 1, n):
            d = hamming(ha, kept[b]["hash"])
            if d <= args.threshold:
                uf.union(a, b)
                near_pairs.append((a, b, d))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(kept[i])
    groups = list(comp.values())
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"  {len(near_pairs)} near-duplicate pairs -> {len(groups)} groups "
          f"(largest {sizes[0]}, singletons {sizes.count(1)})")
    if sizes[0] > 0.2 * n:
        print(f"  WARNING: one group holds {sizes[0]} images ({100 * sizes[0] / n:.0f}%). "
              f"Try a lower --threshold (e.g. {max(args.threshold - 2, 1)}).")

    # ---- 3. assign
    gsplit = assign_splits(groups, ratios, args.seed)
    rows = []
    for gi, g in enumerate(groups):
        for it in g:
            it["group"] = gi
            it["split"] = gsplit[gi]
            rows.append(it)
    rows.sort(key=lambda r: (r["split"], r["class"], r["path"]))

    # ---- 4. verify no leakage
    split_of = {id(it): it["split"] for it in kept}
    leaks = sum(1 for a, b, _ in near_pairs if split_of[id(kept[a])] != split_of[id(kept[b])])

    # ---- outputs
    with open(os.path.join(args.out, "split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "group", "split"])
        for r in rows:
            w.writerow([r["path"], r["class"], r["group"], r["split"]])

    cross = [(kept[a], kept[b], d) for a, b, d in near_pairs if kept[a]["class"] != kept[b]["class"]]
    save_cross_class_sheet(root, cross, os.path.join(args.out, "cross_class_pairs.png"))

    classes = sorted({r["class"] for r in rows})
    splits = list(ratios)
    L = ["LEAKAGE-FREE GROUP SPLIT", "=" * 70,
         f"Source folder      : {root}",
         f"Images read        : {len(items)}",
         f"Exact dups removed : {len(removed_dup)}",
         f"Conflict removed   : {len(removed_conflict)}  (same file, different labels)",
         f"Images kept        : {len(rows)}",
         f"dHash threshold    : {args.threshold}   seed: {args.seed}",
         f"Near-dup groups    : {len(groups)}  (largest {sizes[0]}, singletons {sizes.count(1)})",
         f"Group size counts  : {dict(sorted(Counter(sizes).items()))}",
         f"Cross-class pairs  : {len(cross)}  (see cross_class_pairs.png)",
         f"Leaking pairs      : {leaks}  (must be 0)", ""]
    w0 = max(len(c) for c in classes) + 2
    L.append(f"{'Class':<{w0}}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>8}")
    for c in classes:
        cnt = [sum(1 for r in rows if r["class"] == c and r["split"] == s) for s in splits]
        L.append(f"{c:<{w0}}" + "".join(f"{v:>8}" for v in cnt) + f"{sum(cnt):>8}")
    tot = [sum(1 for r in rows if r["split"] == s) for s in splits]
    L.append(f"{'TOTAL':<{w0}}" + "".join(f"{v:>8}" for v in tot) + f"{sum(tot):>8}")
    L.append(f"{'%':<{w0}}" + "".join(f"{100 * v / len(rows):>7.1f}%" for v in tot))
    if removed_conflict:
        L += ["", "Removed (same file, different labels):"] + [f"  {r['path']}" for r in removed_conflict]
    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(args.out, "split_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    if args.copy:
        for r in rows:
            dst = os.path.join(args.out, r["split"], r["class"])
            os.makedirs(dst, exist_ok=True)
            base = os.path.basename(r["path"])
            prefix = os.path.dirname(r["path"]).split(os.sep)[0].replace(" ", "_")
            shutil.copy2(os.path.join(root, r["path"]), os.path.join(dst, f"{prefix}__{base}"))
        print(f"\nCopied images into {args.out}\\<split>\\<class>\\")
    print(f"\nSaved split.csv and split_summary.txt to {args.out}")


if __name__ == "__main__":
    main()
