# Metaheuristic-Optimized Extreme Learning Machines for Sleep-Disordered Breathing Screening

Reproducibility code for the paper **"Early Screening of Sleep-Disordered Breathing Using Metaheuristic-Optimized Extreme Learning Machines"** (Thaher, Sheta, Ashqar, Chantar, Surani), submitted to *Diagnostics* (MDPI).

> 📄 Paper link / DOI: _to be added on publication._

The study trains a lightweight **Extreme Learning Machine (ELM)** whose hidden-layer weights and biases are optimized by **metaheuristic algorithms** (via the [IntelELM](https://github.com/thieu1995/IntelELM) framework over [MEALPY](https://github.com/thieu1995/mealpy) optimizers), and benchmarks it against eight conventional classifiers for early screening of obstructive sleep apnea (OSA) / sleep-disordered breathing (SDB) from structured clinical predictors.

## Repository layout

```
.
├── config.py, data_utils.py, models.py, experiment.py, reporting.py
├── main_baselines.ipynb            # Baseline pipeline entry point (8 classifiers)
├── preprocess_kaggle_data.ipynb    # Builds the processed SDB dataset from the raw Kaggle file
├── ELM_MHs Framework/              # Metaheuristic-optimized ELM pipeline
│   ├── config_mha.py, data_mha.py, experiment_mha.py
│   ├── optimizers_config.py        # Which optimizers run (ACTIVE_OPTIMIZERS)
│   ├── evaluation_mha.py, plots_mha.py
│   └── main_mha.ipynb              # MHA-ELM pipeline entry point
├── Dataset/
│   └── sdb_dataset.csv             # Processed SDB data (OSA data is NOT included — see below)
├── requirements.txt
└── LICENSE
```

## Datasets

| Dataset | Samples | Features | Availability |
|---|---|---|---|
| **SDB** | 500 | 10 | **Included** (`Dataset/sdb_dataset.csv`). Derived from the public Kaggle dataset [ziya07/sleep-disordered-breathing-detection](https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection). |
| **OSA** | 274 | 31 | **Not included.** Clinical data available from the corresponding author on reasonable request, subject to data-sharing restrictions. |

> **Reproducibility scope:** the study uses **two** datasets, but only **SDB** is publicly redistributable. The SDB experiments therefore reproduce **end-to-end from this repository** (data + code). The **OSA** experiments require the restricted OSA dataset, which is **not** included here — obtain `OSA-data.csv` from the corresponding author (see [Data availability & ethics](#data-availability--ethics)), place it in `Dataset/`, and point the configs at it.

`Dataset/sdb_dataset.csv` is the model-ready file (10 numeric predictors + binary `Diagnosis_of_SDB`, where Mild/Moderate/Severe → 1 and None → 0). To regenerate it from scratch, download the raw file from the Kaggle link above and run `preprocess_kaggle_data.ipynb`.

To run the **OSA** experiments, obtain `OSA-data.csv` from the corresponding author, place it in `Dataset/`, and point the configs at it (see below). It is git-ignored so it cannot be committed accidentally.

## Installation

Developed with **Python 3.9** (Anaconda).

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

## Reproducing the experiments

Both pipelines run on the **SDB** dataset out of the box and repeat each experiment over 20 stratified 80/20 splits, writing per-run and aggregated metrics (accuracy, precision, recall, F1, ROC-AUC, confusion-matrix counts, timings) plus ROC and convergence figures.

**1. Baseline classifiers** — run from the repository root:
```bash
jupyter notebook main_baselines.ipynb     # Run All
# outputs -> Results/
```
Models and hyperparameters are defined in `models.py`; run settings (test size, number of runs, seeds, data path) in `config.py`.

**2. Metaheuristic-optimized ELM** — run from inside the `ELM_MHs Framework/` folder (its modules import by bare name):
```bash
cd "ELM_MHs Framework"
jupyter notebook main_mha.ipynb           # Run All
# outputs -> Results_MHA/  (written at the repository root)
```
ELM/optimizer settings (hidden size, activation, objective, search bounds, runs) are in `config_mha.py`; the set of optimizers that run is controlled by `ACTIVE_OPTIMIZERS` in `optimizers_config.py`.

**Switching to the OSA dataset:** set `DATA_PATH`/`TARGET_COL` in `config.py` and `DATA_PATH` in `ELM_MHs Framework/config_mha.py` to your `Dataset/OSA-data.csv` (target column `class`).

## Models

- **Baselines (8):** Logistic Regression, k-Nearest Neighbours, Decision Tree, Random Forest, SVM (RBF), MLP, XGBoost, and a standard ELM. Hyperparameters are fixed (no tuning search); see `models.py`.
- **Metaheuristic-optimized ELM:** the hidden-layer weights/biases are searched with metaheuristics from the evolutionary, math-based, physics-based, and swarm families — GA, RUN, MEO, CL-PSO, HI-WOA, GWO, HGS, HHO, SeaHO, MGO, and the hybrid GWO-WOA. Each optimizer uses 100 iterations, population size 30, cross-entropy objective, and a [-1, 1] search space; 20 independent runs.

## Citation

```bibtex
@article{thaher2026elmsdb,
  title   = {Early Screening of Sleep-Disordered Breathing Using Metaheuristic-Optimized Extreme Learning Machines},
  author  = {Thaher, Thaer and Sheta, Alaa and Ashqar, Huthaifa I. and Chantar, Hamouda and Surani, Salim},
  journal = {Diagnostics},
  year    = {2026},
  note    = {MDPI}
}
```

## Data availability & ethics

**SDB dataset.** Derived from the public Kaggle dataset *Sleep Disordered Breathing Detection* by Kaggle user `ziya07` — <https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection> (accessed 2025-12-13). The included `Dataset/sdb_dataset.csv` is a **processed subset**: 10 structured predictors plus a binary `Diagnosis_of_SDB` label (Mild/Moderate/Severe → 1, None → 0); the preprocessing is in `preprocess_kaggle_data.ipynb`. Please credit the original Kaggle source and comply with its license and terms of use when reusing this data. If in doubt, download the raw data directly from Kaggle and regenerate the processed file.

**OSA dataset.** Clinical data — **not** redistributed in this repository. It is available from the corresponding author on reasonable request, subject to applicable data-sharing and ethics restrictions.

## License

Code is released under the [MIT License](LICENSE). Dataset licenses follow their respective sources.
