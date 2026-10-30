# Verification performed

## Actual notebook execution

Executed all six code cells of `Model2_corrected.ipynb`, in order, in a fresh Python process using an IPython execution shell. This was actual computation, not only syntax inspection. Displayed tables, stdout and rendered Matplotlib figures were captured into the notebook. Execution counts are 1–6, with no error outputs. No original notebook file was edited.

All 198,487 rows were used. All 60 model/fold fits completed (five existing configurations × 12 folds), followed by the full-data balanced Random Forest refit. No sampled/substitute results. Each row has exactly one OOF score per configuration. No tower occurs in two test folds, so each row's tower was excluded from its training fold. No Logistic Regression convergence error occurred.

Environment: Linux, Python 3.12.14; exact package versions are in `results/metadata.json` and the requirements files. TensorFlow CPU 2.21.0. No Windows machine or interactive Jupyter frontend was available for testing. Windows/Anaconda instructions are provided, but Windows execution is not claimed.

Non-fatal runtime warnings: pandas notes future string/object selection behavior; the pinned version includes string columns in the categorical selector. Keras notes that its `shuffle` argument is ignored for tf.data input; the training Dataset is explicitly shuffled before batching, so training shuffle is active. Neither warning represents skipped training or failed execution. ANN numerical results may vary slightly across hardware/library builds.

## Artifact integrity

- Uploaded and packaged XLSX hashes match exactly.
- Python modules and all corrected notebook cells pass syntax parsing.
- Saved inference pipeline was reloaded; scores match the in-memory fitted pipeline.
- Recomputed every reported OOF accuracy, precision, recall, F1, mean fold accuracy, fold standard deviation and confusion matrix from exported row-level predictions; all match.
- 198,487 distinct row IDs, 12 folds, finite scores in [0,1], one fold per tower verified.

## Streamlit AppTest — passed

- All six navigation pages render without uncaught exceptions.
- Region filtering returns 81,913 western-region rows; clearing filters displays an understandable empty-state message.
- Precision/recall/F1 comparison selections work.
- All five model-report selections work.
- ANN fold selection works (including fold 12).
- Single-record scoring produces score, predicted class and band.
- Batch-mode navigation works; actual batch scoring was tested directly through the same `score_batch` function used by the upload flow.
- A valid real-data batch matches direct pipeline inference.
- Nine invalid cases are rejected: missing column, null, unknown category, negative repair_time, fractional day, invalid calendar date, unobserved region/depot/tech combination, out-of-range repair_time and empty batch.
- Missing metadata, model and dataset files produce messages without uncaught exceptions.
- Missing report/confusion-matrix sections produce a warning without the old split-marker IndexError.

Reproduce with `python tests/test_project.py` after installing dashboard requirements.

## Local server smoke test

Started the actual Streamlit server in headless mode. The health endpoint returned `ok`, and the root page returned HTTP 200. The server was then stopped.

## Remaining verification limits

AppTest validates application logic and Streamlit element state; it is not a visual browser screenshot review. Browser upload/download clicks were not automated; the upload handler's underlying parsing/scoring logic and navigation were exercised as described. No Windows browser, mobile layout, external deployment, forecasting validity, probability calibration, synthetic-data generator or independent real-world test set was tested. No operational reliability claim follows from these results.
