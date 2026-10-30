# TowerScope — corrected Model2 notebook and local dashboard

Start with `REVIEW_SI.md` for the Sinhala review. This package analyzes **recorded site-down status**, not future failure probability. The supplied synthetic data has no future horizon. `repair_time` and assigned `techs` are retrospective inputs. Use only for research/demo exploration, not advance outage warnings.

## Package contents

- `Model2_corrected.ipynb`: corrected notebook with executed outputs.
- `core.py`: complete preprocessing, fold-fitted association selector and input validation. Required beside the notebook and app.
- `app.py`: local Streamlit dashboard; no training on interaction.
- `data/simulated_year7.xlsx`: unchanged supplied dataset.
- `data/records.csv.gz`: normalized data for cached exploration.
- `data/example_batch.csv`: real example inputs from the supplied dataset.
- `results/`: actual OOF metrics, reports, confusion matrices, fold histories/selections, per-row OOF scores, metadata and full-data inference pipeline.
- `requirements.txt`: dashboard dependencies; TensorFlow is not required to run the dashboard.
- `requirements-notebook.txt`: additional dependencies for notebook retraining.
- `REVIEW_SI.md`, `TESTING.md`: corrections, limitations and verification.

## Windows / Anaconda — run the dashboard

1. Extract the entire ZIP. Keep all files and folders together. Open **Anaconda Prompt**.
2. Change to the folder that actually contains `app.py` and `requirements.txt` (avoid the extra nested-folder issue):

```bat
cd /d "C:\Users\USER\Desktop\telecom-site-down-analysis"
dir app.py requirements.txt
```

Replace the path with your extracted folder. If `dir` cannot find both files, locate them with `dir /s /b app.py` and change to its parent folder.

3. Use a new Python **3.12** environment. The old Python 3.11 `oral-dashboard` environment does not match these package versions:

```bat
conda create -n tower-dashboard python=3.12 -y
conda activate tower-dashboard
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal open. Press Ctrl+C to stop.

If pip reports `No matching distribution`, first verify `python --version` prints 3.12.x. If a configured mirror is missing packages, use the public index:

```bat
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

Do not change scikit-learn versions independently of the saved pipeline. If you change training dependencies, rerun the notebook to regenerate the model and results.

## Read or rerun the notebook

The notebook contains actual saved execution outputs. Opening it does not require training again. To rerun:

```bat
conda activate tower-dashboard
cd /d "C:\Users\USER\Desktop\telecom-site-down-analysis"
python -m pip install -r requirements-notebook.txt
python -m jupyter lab Model2_corrected.ipynb
```

Stop the Streamlit app before retraining and restart it after successful completion so its caches load the new results. Select the environment's Python kernel and **Restart Kernel and Run All**. Start Jupyter from this folder so `core.py` and `data/` resolve. Full execution trains 5 configurations over 12 folds plus the final RF refit; it can take substantial time and several GB of memory. All input rows are used; no hidden sample or skipped model. Do not interrupt midway and then combine partially updated results with older files. The shipped result bundle is complete.

## Reading results correctly

Four families, five existing configurations: Logistic Regression; ANN; Decision Tree; RF with 20 trees; RF with 200 trees and balanced classes. Estimator hyperparameters are unchanged. Pipeline corrections change representation and evaluation: categorical one-hot encoding, training-only association selection/imputation/scaling, and tower-disjoint folds.

`accuracy` is pooled out-of-fold accuracy. `mean_fold_accuracy` and its descriptive standard deviation are separate. Class 1 recall/precision/F1 matter because 69.95% of records are class 0. Reports include every OOF row, not only the last fold. No claim that accuracy proves correctness.

The deployed model is the original balanced 200-tree configuration refitted on all rows. It is explicitly not selected as “best” from these results. Scores are uncalibrated. Original score-band cutoffs 0.4/0.7 are kept only as illustrative labels.

Dataset filters apply only to the Dataset explorer. They do not alter saved model metrics. No cached metric is presented as a filtered-model evaluation.

## Batch scoring

Use the provided example CSV. Required raw columns:
`Region, Depot, Network_type, Month, equipment_vendor, Day, techs, repair_time`.

Use names and original technician IDs, not manual numeric category codes. `nothern` is accepted and normalized to `northern`; month case is normalized. Inputs must be complete, use known categories, observed region/depot/technician combinations, valid non-leap month/day values and a repair_time within the observed training range. Additional columns are retained in the output but not used as features. Maximum 200,000 rows / 30 MB per upload.

## Troubleshooting

- Missing data/result/model file: restore the entire ZIP, or run the corrected notebook to regenerate results. The app shows a message instead of parsing absent text markers.
- `ModuleNotFoundError: core`: keep `core.py` alongside `app.py` and the notebook.
- `ModuleNotFoundError: streamlit`: activate `tower-dashboard` and install requirements with `python -m pip`.
- Port busy: `python -m streamlit run app.py --server.port 8502`.
- Run tests: `python tests/test_project.py` from the project folder, after installing dashboard requirements.

The original uploaded notebook was not modified. No external hosting or GitHub upload was performed.
