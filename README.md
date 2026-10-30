# Telecom Site-Down Status Classification and Analysis

**TowerScope** is a Python machine-learning project with a blue-and-white Streamlit dashboard for exploring synthetic telecom records and classifying their recorded site-down status.

The target is `is_site_down` (`0` = not down, `1` = down). This is retrospective classification: observed repair duration and assigned technician may be post-incident inputs. It does not estimate failures per operating hour or forecast future outages.

## What is included

- Corrected, executed Jupyter notebook and shared preprocessing code.
- Logistic Regression, ANN, Decision Tree and two existing Random Forest configurations.
- Training-fold-only association selection, categorical encoding, imputation and scaling.
- Twelve tower-disjoint cross-validation folds with complete out-of-fold results.
- Saved full-data Random Forest pipeline for single-record and batch scoring.
- Six dashboard pages: Overview, Dataset explorer, Model comparison, Model reports, ANN learning curves and Site status scoring.
- Input validation, cached data/results and automated dashboard checks.

## Run locally

Use a clean **Python 3.12** environment. From the repository root in Anaconda Prompt:

```bat
conda create -n tower-dashboard-clean python=3.12 pip -y
conda activate tower-dashboard-clean
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit. Saved models and results are included; no retraining is needed to run the dashboard. Full Windows instructions, batch inputs and troubleshooting: [SETUP.md](SETUP.md).

## Data

The user-supplied `simulated_year7.xlsx` contains **198,487 records**, **6,773 towers** and **59,645 down-labelled records (30.05%)**. Records can repeat the same tower; this percentage is not the percentage of unique towers that failed. No missing cells or exact duplicate rows were found. There are 13,929 repeated tower/month/day combinations, retained within tower groups because no event identifier was supplied.

Dataset provenance and distribution status: [data/README.md](data/README.md).

## Verified results

All records have one held-out prediction per model configuration. These metrics describe the supplied synthetic dataset, using the same 12 tower-disjoint folds. Precision, recall and F1 below refer to class 1.

| Configuration | OOF accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.80% | 70.71% | 33.22% | 45.21% |
| Decision Tree | 74.07% | 66.44% | 27.70% | 39.10% |
| Random Forest (20) | 71.30% | 76.36% | 6.49% | 11.97% |
| Random Forest (balanced 200) | 73.22% | 56.36% | 48.21% | 51.97% |
| ANN | 73.66% | 80.31% | 16.35% | 27.17% |

The always-class-0 reference accuracy is 69.95%. Accuracy alone does not establish model usefulness. The original balanced 200-tree RF is retained for inference; it is not labelled the best model. Scores are uncalibrated, and the original 0.4/0.7 score bands are illustrative. Dataset-explorer filters do not change saved global model metrics.

## Project files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `core.py` | Shared selection, preprocessing and input validation |
| `Model2_corrected.ipynb` | Executed analysis and training workflow |
| `data/` | Supplied XLSX, normalized records and batch example |
| `results/` | OOF scores, reports, histories, metadata and inference model |
| `tests/test_project.py` | Metrics, input and dashboard checks |
| `requirements.txt` | Dashboard dependencies |
| `requirements-notebook.txt` | Notebook retraining dependencies |
| `REVIEW_SI.md` | Sinhala review and correction notes |
| `TESTING.md` | Execution evidence and test boundaries |
| `GITHUB_UPLOAD_SI.md` | GitHub upload instructions |

## Reproduce or test

```bat
python tests/test_project.py
```

To retrain, install `requirements-notebook.txt`, launch Jupyter from this folder and run the notebook top to bottom. Training includes 60 cross-validation fits plus a final RF fit and requires several GB of RAM. Stop the app before regenerating results; restart after successful completion.

The included GitHub Actions workflow runs dashboard checks without retraining. Local tests passed; a GitHub-hosted run has not yet been executed. See [TESTING.md](TESTING.md).

## Interpretation and scope

Synthetic generation code, independent real-world validation, event timestamps and a future target horizon were not supplied. Results measure held-out-tower classification within the supplied synthetic period. They do not demonstrate calibrated future-failure probabilities or operational reliability. Association selection is exploratory and does not establish causation.

## License

No software or dataset license has been assigned in this package. The repository owner should choose an appropriate software license and confirm dataset redistribution rights before granting reuse rights. See the data provenance note for what is known about the supplied file.
