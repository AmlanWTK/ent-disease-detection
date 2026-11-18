"""
download_papers.py - Download open-access papers that used the public ear datasets.

Usage (from the repo root, Windows):
    python scripts\\download_papers.py

Saves PDFs to papers/ear/<dataset>/ and writes papers/ear/README.md listing
what was downloaded and which paywalled papers you need to get manually
(through your university library, ResearchGate, or by emailing the authors).
PDFs are git-ignored; only the README is pushed.
"""

import os
import sys
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers", "ear")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# (dataset folder, file name, title, year, open-access PDF url or None, landing page / DOI)
PAPERS = [
    # ---- Chile Ear Imagery Database (880 images)
    ("chile_880", "2020_Viscaino_PLOSONE_SVM.pdf",
     "Computer-aided diagnosis of external and middle ear conditions: A machine learning approach", 2020,
     "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0229226&type=printable",
     "https://doi.org/10.1371/journal.pone.0229226"),
    ("chile_880", "2021_Ucar_IRBM_HypercolumnBiLSTM.pdf",
     "Classification of Different Tympanic Membrane Conditions Using Fused Deep Hypercolumn Features and Bi-LSTM", 2021,
     None, "https://doi.org/10.1016/j.irbm.2021.01.001"),
    ("chile_880", "2022_Mohammed_JDigitImaging_CNN-LSTM.pdf",
     "Classification of Ear Imagery Database using Bayesian Optimization based on CNN-LSTM Architecture", 2022,
     None, "https://doi.org/10.1007/s10278-022-00617-8"),
    ("chile_880", "2023_Afify_AnnBiomedEng_BayesCNN.pdf",
     "Insight into Automatic Image Diagnosis of Ear Conditions Based on Optimized Deep Learning Approach", 2023,
     "https://link.springer.com/content/pdf/10.1007/s10439-023-03422-8.pdf",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC10940396/"),
    ("chile_880", "2024_Akyol_MTAP_SoftVotingEnsemble.pdf",
     "An ensemble approach for classification of tympanic membrane conditions using soft voting classifier", 2024,
     "https://link.springer.com/content/pdf/10.1007/s11042-024-18631-z.pdf",
     "https://doi.org/10.1007/s11042-024-18631-z"),
    ("chile_880", "2024_Comert_NCA_ViT-SVM.pdf",
     "Computerized otoscopy image-based AI model utilizing deep features provided by vision transformer, grid search and SVM", 2024,
     None, "https://doi.org/10.1007/s00521-024-10457-y"),
    ("chile_880", "2025_Demircan_NCA_ViT-kNN.pdf",
     "Enhancing intra-aural disease classification with attention-based deep learning models", 2025,
     None, "https://doi.org/10.1007/s00521-025-10990-4"),
    ("chile_880", "2025_Tatli_DergiPark_CNN-comparison.pdf",
     "Classification of ear pathologies using deep learning on otoscopic images", 2025,
     "https://dergipark.org.tr/en/download/article-file/4874185",
     "https://dergipark.org.tr/en/download/article-file/4874185"),
    ("chile_880", "2025_Ndubuisi_arXiv_DataLeakage.pdf",
     "Validating Vision Transformers for Otoscopy: Performance and Data-Leakage Effects", 2025,
     "https://arxiv.org/pdf/2511.04872", "https://arxiv.org/abs/2511.04872"),
    # ---- Turkiye eardrum.zip (282 images)
    ("turkey_282", "2020_Basaran_BSPC_FasterRCNN-VGG16.pdf",
     "Convolutional neural network approach for automatic tympanic membrane detection and classification", 2020,
     None, "https://doi.org/10.1016/j.bspc.2019.101734"),
    ("turkey_282", "2021_Tsutsumi_OtolNeurotol_MobileNetV2.pdf",
     "A web-based deep learning model for automated diagnosis of otoscopic images", 2021,
     "https://escholarship.org/content/qt1x2850v1/qt1x2850v1.pdf",
     "https://escholarship.org/uc/item/1x2850v1"),
    # ---- Turkiye Eardrum Dataset (956 images, Kaggle)
    ("turkey_956", "2019_Comert_FusedDeepFeatures_SVM.pdf",
     "Fusing fine-tuned deep features for recognizing different tympanic membranes", 2019,
     None, "https://acikerisim.samsun.edu.tr/handle/123456789/798"),
    ("turkey_956", "2021_Alhudhaif_PeerJCS_CBAM-CNN.pdf",
     "Otitis media detection using tympanic membrane images with a novel multi-class machine learning algorithm", 2021,
     "https://peerj.com/articles/cs-405.pdf", "https://doi.org/10.7717/peerj-cs.405"),
    ("turkey_956", "2022_Caliskan_Kocaeli_VGG16-SVM.pdf",
     "Classification of Tympanic Membrane Images based on VGG16 Model", 2022,
     "https://dergipark.org.tr/en/download/article-file/2284066",
     "https://dergipark.org.tr/en/download/article-file/2284066"),
    ("turkey_956", "2026_Seker_ICCPEAS_SupCon-ResNet50.pdf",
     "Enhanced Feature Representation for Otoscopic Image Classification Using Supervised Contrastive Learning", 2026,
     "https://www.atlantis-press.com/article/126024219.pdf",
     "https://www.atlantis-press.com/article/126024219.pdf"),
]


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF (probably a login, captcha or landing page)")
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def main():
    ok, failed, manual = [], [], []
    for folder, fname, title, year, pdf_url, page in PAPERS:
        d = os.path.join(ROOT, folder)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, fname)
        rel = f"{folder}/{fname}"
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  [exists]  {rel}")
            ok.append((rel, title, year, page))
            continue
        if not pdf_url:
            print(f"  [manual]  {rel}  (paywalled) -> {page}")
            manual.append((rel, title, year, page))
            continue
        try:
            size = download(pdf_url, dest)
            print(f"  [ok]      {rel}  ({size / 1024:.0f} KB)")
            ok.append((rel, title, year, page))
        except Exception as e:  # noqa: BLE001
            print(f"  [failed]  {rel}: {e}\n            open in browser: {page}")
            failed.append((rel, title, year, page))
        time.sleep(1)

    lines = ["# Ear dataset papers", "",
             "Run `python scripts/download_papers.py` to (re)download. PDFs are git-ignored.", "",
             f"## Downloaded ({len(ok)})", ""]
    lines += [f"- `{r}` ({y}) - {t}" for r, t, y, _ in ok]
    lines += ["", f"## Download failed - open in a browser and save with this file name ({len(failed)})", ""]
    lines += [f"- `{r}` ({y}) - {t}: {p}" for r, t, y, p in failed]
    lines += ["", f"## Paywalled - get via library / ResearchGate / authors ({len(manual)})", ""]
    lines += [f"- `{r}` ({y}) - {t}: {p}" for r, t, y, p in manual]
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nDone: {len(ok)} downloaded, {len(failed)} failed, {len(manual)} paywalled.")
    print(f"Folder: {ROOT}")
    print("See papers/ear/README.md for the manual list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
