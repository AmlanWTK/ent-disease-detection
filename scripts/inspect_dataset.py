"""
inspect_dataset.py - Full inspection of an image classification dataset.

Usage (Windows):
    pip install pillow numpy matplotlib
    python inspect_dataset.py "E:\\dataset\\Datos"

Optional:
    python inspect_dataset.py "E:\\dataset\\Datos" --out "E:\\dataset\\report" --no-plots

What it reports:
  1. Folder tree (with file counts)
  2. File types found (images vs. other files)
  3. Splits (e.g. Training / Testing) and classes (sub-folders) with counts
  4. Image parameters: format, color mode, channels, width/height, aspect ratio,
     file size, DPI, EXIF presence
  5. Pixel statistics: per-channel mean/std (for normalization), brightness
  6. Problems: corrupt/unreadable images, exact duplicates (MD5),
     near-duplicates (perceptual hash), duplicates across splits (data leakage)
  7. Saves: report.txt, per_image.csv, summary.json and charts (PNG)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

try:
    import numpy as np
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Missing packages. Run:  pip install pillow numpy matplotlib")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
SPLIT_WORDS = {"train", "training", "test", "testing", "val", "valid", "validation",
               "entrenamiento", "prueba", "validacion", "training-validation"}


# ----------------------------------------------------------------- helpers
def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(img, size=8):
    """Perceptual difference hash (64-bit int). Similar images -> similar hashes."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def stats(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {"min": float(arr.min()), "max": float(arr.max()),
            "mean": round(float(arr.mean()), 2), "median": float(np.median(arr)),
            "std": round(float(arr.std()), 2)}


def guess_split_and_class(rel_parts):
    """rel_parts = folders between root and the file.
    Class = deepest folder. Split = a higher folder whose name looks like a split."""
    if not rel_parts:
        return "(root)", "(root)"
    cls = rel_parts[-1]
    split = "(none)"
    for p in rel_parts[:-1]:
        if p.lower() in SPLIT_WORDS or any(w in p.lower() for w in ("train", "test", "val")):
            split = p
            break
    if split == "(none)" and len(rel_parts) > 1:
        split = rel_parts[0]
    return split, cls


def print_tree(root, out, max_depth=4):
    out.append(f"{os.path.basename(root) or root}/")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = os.path.relpath(dirpath, root).count(os.sep) + (0 if dirpath == root else 1)
        if dirpath == root:
            continue
        if depth > max_depth:
            continue
        indent = "    " * depth
        exts = Counter(os.path.splitext(f)[1].lower() or "(no ext)" for f in filenames)
        ext_txt = ", ".join(f"{v} {k}" for k, v in exts.most_common()) if filenames else "no files"
        out.append(f"{indent}{os.path.basename(dirpath)}/  [{ext_txt}]")


# ----------------------------------------------------------------- main scan
def scan(root, max_pixel_images, near_dup_threshold):
    records, other_files, corrupt = [], [], []
    ext_counter = Counter()

    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            all_files.append(os.path.join(dirpath, f))
    all_files.sort()
    total = len(all_files)
    print(f"Found {total} files. Reading images...")

    # per-channel running sums for mean/std (on images resized to 128x128 RGB)
    ch_sum = np.zeros(3)
    ch_sq = np.zeros(3)
    px_count = 0
    pixel_imgs = 0

    t0 = time.time()
    for i, path in enumerate(all_files, 1):
        ext = os.path.splitext(path)[1].lower()
        ext_counter[ext or "(no ext)"] += 1
        rel = os.path.relpath(path, root)
        if ext not in IMAGE_EXTS:
            other_files.append(rel)
            continue

        parts = rel.split(os.sep)[:-1]
        split, cls = guess_split_and_class(parts)
        rec = {"path": rel, "split": split, "class": cls, "ext": ext,
               "file_size": os.path.getsize(path)}
        try:
            with Image.open(path) as im:
                im.verify()  # detect truncated/corrupt files
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                rec.update({
                    "format": im.format or ext.strip(".").upper(),
                    "mode": im.mode,
                    "channels": len(im.getbands()),
                    "width": im.width,
                    "height": im.height,
                    "aspect": round(im.width / im.height, 3) if im.height else 0,
                    "dpi": str(im.info.get("dpi", "")),
                    "has_exif": bool(im.info.get("exif")),
                })
                rgb = im.convert("RGB")
                rec["dhash"] = dhash(rgb)
                small = np.asarray(rgb.resize((128, 128)), dtype=np.float64) / 255.0
                rec["brightness"] = round(float(small.mean()), 4)
                rec["mean_r"], rec["mean_g"], rec["mean_b"] = (round(float(x), 4) for x in small.mean(axis=(0, 1)))
                if pixel_imgs < max_pixel_images:
                    flat = small.reshape(-1, 3)
                    ch_sum += flat.sum(0)
                    ch_sq += (flat ** 2).sum(0)
                    px_count += flat.shape[0]
                    pixel_imgs += 1
            rec["md5"] = md5_of(path)
            records.append(rec)
        except Exception as e:  # noqa: BLE001
            corrupt.append((rel, str(e)))

        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} files  ({time.time() - t0:.0f}s)", end="\r")
    print()

    mean = (ch_sum / px_count) if px_count else np.zeros(3)
    std = np.sqrt(np.maximum(ch_sq / px_count - mean ** 2, 0)) if px_count else np.zeros(3)

    # exact duplicates
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    exact_dups = [g for g in by_md5.values() if len(g) > 1]

    # near duplicates (O(n^2) but fine for a few thousand images)
    near_dups = []
    n = len(records)
    for a in range(n):
        ha = records[a]["dhash"]
        for b in range(a + 1, n):
            if records[a]["md5"] == records[b]["md5"]:
                continue
            if hamming(ha, records[b]["dhash"]) <= near_dup_threshold:
                near_dups.append((records[a], records[b]))

    return {
        "records": records, "other_files": other_files, "corrupt": corrupt,
        "ext_counter": ext_counter, "mean": mean, "std": std, "pixel_imgs": pixel_imgs,
        "exact_dups": exact_dups, "near_dups": near_dups,
    }


# ----------------------------------------------------------------- report
def build_report(root, res, near_dup_threshold):
    R = res["records"]
    L = []
    add = L.append
    bar = "=" * 70

    add(bar); add(f"DATASET INSPECTION REPORT"); add(bar)
    add(f"Root folder : {root}")
    add(f"Scanned at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_bytes = sum(r["file_size"] for r in R)
    add(f"Images OK   : {len(R)}   ({human_size(total_bytes)})")
    add(f"Corrupt     : {len(res['corrupt'])}")
    add(f"Other files : {len(res['other_files'])}")
    add("")

    add("1. FOLDER STRUCTURE"); add("-" * 70)
    tree = []
    print_tree(root, tree)
    L.extend(tree[:200])
    if len(tree) > 200:
        add(f"... ({len(tree) - 200} more folders)")
    add("")

    add("2. FILE TYPES"); add("-" * 70)
    for ext, c in res["ext_counter"].most_common():
        kind = "image" if ext in IMAGE_EXTS else "other"
        add(f"  {ext:<10} {c:>7}   ({kind})")
    if res["other_files"]:
        add("  Non-image files (first 20):")
        for f in res["other_files"][:20]:
            add(f"    - {f}")
    add("")

    add("3. SPLITS AND CLASSES"); add("-" * 70)
    split_class = defaultdict(Counter)
    for r in R:
        split_class[r["split"]][r["class"]] += 1
    classes = sorted({r["class"] for r in R})
    splits = sorted(split_class)
    add(f"  Number of classes : {len(classes)}")
    add(f"  Classes           : {classes}")
    add(f"  Splits detected   : {splits}")
    add("")
    w = max([len(c) for c in classes] + [10])
    header = f"  {'Class':<{w}} " + "".join(f"{s[:14]:>16}" for s in splits) + f"{'TOTAL':>10}{'%':>8}"
    add(header)
    for c in classes:
        row = [split_class[s][c] for s in splits]
        tot = sum(row)
        add(f"  {c:<{w}} " + "".join(f"{v:>16}" for v in row) + f"{tot:>10}{100 * tot / max(len(R), 1):>7.1f}%")
    totals = [sum(split_class[s].values()) for s in splits]
    add(f"  {'TOTAL':<{w}} " + "".join(f"{v:>16}" for v in totals) + f"{len(R):>10}")
    counts = Counter(r["class"] for r in R)
    if counts:
        ratio = max(counts.values()) / max(min(counts.values()), 1)
        add(f"\n  Class imbalance ratio (largest/smallest): {ratio:.2f}"
            + ("  -> BALANCED" if ratio < 1.5 else "  -> IMBALANCED (consider class weights / augmentation)"))
    add("")

    add("4. IMAGE PARAMETERS"); add("-" * 70)
    add(f"  Formats      : {dict(Counter(r['format'] for r in R))}")
    add(f"  Color modes  : {dict(Counter(r['mode'] for r in R))}")
    add(f"  Channels     : {dict(Counter(r['channels'] for r in R))}")
    sizes = Counter(f"{r['width']}x{r['height']}" for r in R)
    add(f"  Unique sizes : {len(sizes)}")
    for s, c in sizes.most_common(10):
        add(f"      {s:<14} {c} images")
    for key, label in [("width", "Width (px)"), ("height", "Height (px)"),
                       ("aspect", "Aspect W/H"), ("file_size", "File size (B)")]:
        s = stats([r[key] for r in R])
        if s:
            add(f"  {label:<14}: min {s['min']:.0f}  max {s['max']:.0f}  mean {s['mean']}  median {s['median']}"
                if key != "aspect" else
                f"  {label:<14}: min {s['min']}  max {s['max']}  mean {s['mean']}  median {s['median']}")
    add(f"  DPI values   : {dict(Counter(r['dpi'] for r in R).most_common(5))}")
    add(f"  With EXIF    : {sum(r['has_exif'] for r in R)} / {len(R)}")
    add("")
    add("  Per-class image size summary:")
    for c in classes:
        cr = [r for r in R if r["class"] == c]
        ws, hs = stats([r["width"] for r in cr]), stats([r["height"] for r in cr])
        add(f"    {c:<{w}} width {ws['min']:.0f}-{ws['max']:.0f}  height {hs['min']:.0f}-{hs['max']:.0f}  "
            f"avg size {human_size(np.mean([r['file_size'] for r in cr]))}")
    add("")

    add("5. PIXEL STATISTICS (images resized to 128x128, RGB, scale 0-1)"); add("-" * 70)
    m, s = res["mean"], res["std"]
    add(f"  Computed on {res['pixel_imgs']} images")
    add(f"  Mean (R,G,B) : [{m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f}]")
    add(f"  Std  (R,G,B) : [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]")
    add("  -> use in PyTorch: transforms.Normalize(mean=[...], std=[...])")
    add("  Mean brightness per class:")
    for c in classes:
        b = [r["brightness"] for r in R if r["class"] == c]
        add(f"    {c:<{w}} {np.mean(b):.4f}")
    add("")

    add("6. DATA QUALITY PROBLEMS"); add("-" * 70)
    add(f"  Corrupt / unreadable images: {len(res['corrupt'])}")
    for f, e in res["corrupt"][:20]:
        add(f"    - {f}: {e}")
    n_exact = sum(len(g) - 1 for g in res["exact_dups"])
    add(f"  Exact duplicate files (same bytes): {n_exact} extra copies in {len(res['exact_dups'])} groups")
    for g in res["exact_dups"][:10]:
        add("    - " + "  ==  ".join(r["path"] for r in g))
    nd = res["near_dups"]
    add(f"  Near-duplicate pairs (dHash distance <= {near_dup_threshold}): {len(nd)}")
    imgs_in_nd = {r["path"] for pair in nd for r in pair}
    add(f"  Images involved in near-duplicates: {len(imgs_in_nd)} "
        f"({100 * len(imgs_in_nd) / max(len(R), 1):.1f}% of dataset)")
    for a, b in nd[:15]:
        add(f"    - {a['path']}  ~  {b['path']}")
    cross_class = [(a, b) for a, b in nd if a["class"] != b["class"]]
    add(f"  Near-duplicates with DIFFERENT class labels: {len(cross_class)}  (possible label noise)")
    for a, b in cross_class[:10]:
        add(f"    - [{a['class']}] {a['path']}  ~  [{b['class']}] {b['path']}")
    leak = [(a, b) for a, b in nd if a["split"] != b["split"]]
    leak += [(g[0], g[1]) for g in res["exact_dups"] if g[0]["split"] != g[1]["split"]]
    add(f"  Duplicates ACROSS splits (train/test leakage): {len(leak)}")
    for a, b in leak[:10]:
        add(f"    - [{a['split']}] {a['path']}  ~  [{b['split']}] {b['path']}")
    add("")

    add("7. RECOMMENDATIONS"); add("-" * 70)
    if res["corrupt"]:
        add("  * Remove or re-download the corrupt images listed above.")
    if len(sizes) > 1:
        add(f"  * Images have {len(sizes)} different sizes -> resize to a fixed input size (e.g. 224x224).")
    if leak:
        add("  * Train/test leakage found -> re-split so near-duplicates stay in the same split.")
    if len(imgs_in_nd) > 0.1 * max(len(R), 1):
        add("  * Many near-duplicates -> likely video frames from the same patient; split by patient/group.")
    if counts and max(counts.values()) / max(min(counts.values()), 1) >= 1.5:
        add("  * Classes are imbalanced -> use class weights, oversampling or augmentation.")
    if len(R) < 2000:
        add("  * Small dataset -> use transfer learning (ImageNet weights) + strong augmentation.")
    add("")
    add(bar)
    return "\n".join(L)


def save_outputs(out_dir, root, res, report, make_plots):
    os.makedirs(out_dir, exist_ok=True)
    R = res["records"]

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cols = ["path", "split", "class", "ext", "format", "mode", "channels", "width", "height",
            "aspect", "file_size", "dpi", "has_exif", "brightness", "mean_r", "mean_g", "mean_b", "md5"]
    with open(os.path.join(out_dir, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(R)

    summary = {
        "root": root,
        "n_images": len(R),
        "n_corrupt": len(res["corrupt"]),
        "classes": dict(Counter(r["class"] for r in R)),
        "splits": {s: dict(Counter(r["class"] for r in R if r["split"] == s))
                   for s in sorted({r["split"] for r in R})},
        "sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in R).most_common(20)),
        "modes": dict(Counter(r["mode"] for r in R)),
        "mean_rgb": [round(float(x), 4) for x in res["mean"]],
        "std_rgb": [round(float(x), 4) for x in res["std"]],
        "exact_duplicate_groups": len(res["exact_dups"]),
        "near_duplicate_pairs": len(res["near_dups"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not make_plots or not R:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping charts (pip install matplotlib)")
        return

    # class distribution per split
    classes = sorted({r["class"] for r in R})
    splits = sorted({r["split"] for r in R})
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4.5))
    x = np.arange(len(classes))
    wbar = 0.8 / len(splits)
    for i, s in enumerate(splits):
        vals = [sum(1 for r in R if r["class"] == c and r["split"] == s) for c in classes]
        ax.bar(x + i * wbar, vals, wbar, label=s)
    ax.set_xticks(x + wbar * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Images"); ax.set_title("Images per class"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=120); plt.close(fig)

    # width/height scatter
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter([r["width"] for r in R], [r["height"] for r in R], s=8, alpha=0.5)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)"); ax.set_title("Image dimensions")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "image_sizes.png"), dpi=120); plt.close(fig)

    # brightness per class
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4.5))
    ax.boxplot([[r["brightness"] for r in R if r["class"] == c] for c in classes])
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Mean brightness (0-1)"); ax.set_title("Brightness by class")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "brightness_by_class.png"), dpi=120); plt.close(fig)

    # sample grid: up to 5 images per class
    per = 5
    fig, axes = plt.subplots(len(classes), per, figsize=(per * 2.2, len(classes) * 2.3), squeeze=False)
    for ci, c in enumerate(classes):
        items = [r for r in R if r["class"] == c][:per]
        for j in range(per):
            a = axes[ci][j]; a.axis("off")
            if j < len(items):
                with Image.open(os.path.join(root, items[j]["path"])) as im:
                    a.imshow(ImageOps.exif_transpose(im).convert("RGB"))
            if j == 0:
                a.set_title(c, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "samples_per_class.png"), dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Inspect an image dataset folder.")
    ap.add_argument("root", nargs="?", default=r"E:\dataset\Datos", help="Dataset folder")
    ap.add_argument("--out", default=None, help="Output folder (default: <root>_report)")
    ap.add_argument("--max-pixel-images", type=int, default=5000,
                    help="Max images used for mean/std calculation")
    ap.add_argument("--near-dup-threshold", type=int, default=5,
                    help="dHash Hamming distance for near-duplicates (0-64, lower = stricter)")
    ap.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}")
    out_dir = args.out or (root.rstrip("\\/") + "_report")

    res = scan(root, args.max_pixel_images, args.near_dup_threshold)
    report = build_report(root, res, args.near_dup_threshold)
    print(report)
    save_outputs(out_dir, root, res, report, not args.no_plots)
    print(f"\nSaved report, CSV, JSON and charts to: {out_dir}")


if __name__ == "__main__":
    main()
