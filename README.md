# ENT Disease Detection

Deep learning for detecting diseases from images of the **ear**, **nose** and **mouth**.

## Folder structure

```
ent-disease-detection/
├── data/          # datasets (not pushed to GitHub, see data/README.md)
│   ├── ear/
│   ├── nose/
│   └── mouth/
├── scripts/       # utility scripts (dataset inspection, splitting, etc.)
├── notebooks/     # Jupyter notebooks for experiments
├── src/           # training / model code
├── models/        # saved model weights (not pushed)
└── reports/       # generated reports and charts (not pushed)
```

## Setup

```bash
git clone https://github.com/<your-username>/ent-disease-detection.git
cd ent-disease-detection
pip install -r requirements.txt
```

Download the datasets into `data/` (see `data/README.md`).

## Inspect a dataset

```bash
python scripts/inspect_dataset.py data/ear/chile --out reports/ear_chile
```
