# Model2 — පරීක්ෂාව සහ නිවැරදි කළ වෙනස්කම්

පරීක්ෂා කළ source: `Model2(1).ipynb` (cells 98; පහත cell අංක සියලු markdown/code cells 1 සිට ගණන් කර ඇත). Dataset: `simulated_year7.xlsx`, `Sheet1`. Original notebook සහ dataset වෙනස් කර නැත. `Model2_corrected.ipynb` යනු වෙනම corrected copy එකකි; එහි preprocessing implementation එක `core.py` තුළ ඇත. ZIP එක සම්පූර්ණයෙන් extract කරන්න.

## Dataset එකෙන් තහවුරු කළ දේ

- Rows 198,487; columns 12; unique towers 6,773.
- Missing cells 0; සම්පූර්ණ duplicate rows 0.
- `is_site_down`: class 0 = 138,842; class 1 = 59,645.
- හැමවිටම class 0 කියන reference accuracy = 69.9502%. එබැවින් accuracy එක පමණක් ප්‍රමාණවත් නැත.
- එකම tower/Month/Day සඳහා නැවත ඇති rows 13,929ක් තිබේ. ඒවා සම්පූර්ණ duplicate rows නොවේ. Event ID නැති නිසා ඒවා එකම incident ද යන්න තහවුරු කළ නොහැක. Delete නොකර, එකම tower එක එකම fold තුළ තබා ඇත.
- `April` capitalization සහ `nothern` spelling තිබේ. Month lowercase කර, `nothern` → `northern` ලෙස consistent normalization කර ඇත.
- Future failure target/horizon, exact year, event timestamp, synthetic generation code සහ `repair_time` units ලබා දී නැත.

## තහවුරු කළ code / output ගැටලු

| Original cell | Relevant code | ගැටලුව හා බලපෑම | අවම නිවැරදි කිරීම / දැන් කළ දේ |
|---|---|---|---|
| 18 | `pd.read_excel("c:/Users/USER/Desktop/Today meeting/simulated_year7.xlsx")` | එක Windows machine එකක path එකක්; වෙනත් machine එකක load වන්නේ නැත. | Project-relative `data/simulated_year7.xlsx` සහ පැහැදිලි missing-file error. |
| 23 | `df.dropna()` | Return value assign නොකරන නිසා df වෙනස් නොවේ. මේ dataset එකේ nulls නැති බැවින් saved results වලට එය බලපා නැත. | Missing-value audit; training-fold-only imputers. Inference missing inputs reject කරයි. |
| 40 | `df2['Region']=df2['Region'].map(Region)` | Saved output හි `NameError: df2 is not defined`. Cell 36 පසුව නිසි order එකෙන් run කළොත් මෙය ඇති නොවිය හැක; saved notebook state inconsistent බව තහවුරු වේ. | Clean sequential execution; stale outputs වෙනුවට නව execution outputs. |
| 32 | `relationship_data = {...}` | Hard-coded statistical results වෙනත් data/run එකකට stale විය හැක. | Training data වලින් association values නැවත calculate කර JSON ලෙස save කිරීම. |
| 65 | `model.fit(...)` සහ `history = model.fit(...)` | Logistic Regression එක එකම fold එකේ දෙවරක් fit කරයි. මෙය ANN history එකක් ද නොවේ. | එක fit call එකක් පමණක්. |
| 66–88 | Loop එකෙන් පසු `Y_*_test`, `Y_*_pred`, `history_ANN` | අවසන් fold variables පමණක් ඉතිරි වේ. ඒ reports/plots සමස්ත CV performance නොවේ. Model comparison misinterpret විය හැක. | සෑම row එකකම OOF score, fold ID; pooled report/confusion matrix; ANN fold histories 12ම. |
| 79, 89 | RF `n_estimators=20` පසුව `200, class_weight='balanced'` | වෙනස් configurations එකම variable/name යටතේ overwrite වේ; earlier metrics හා deployed model නොගැළපේ. | Existing configurations දෙකම වෙනම evaluate/report කිරීම. Balanced RF deployment config ලෙස පැහැදිලිව නම් කිරීම. |
| 90 | `joblib.dump((model_RF, scaler), 'RF_model.pkl')` | Last-fold fit එක පමණක් save කරයි; categorical mappings save කර නැත. | Full-data refit කරන selector/encoder/scaler/model සම්පූර්ණ Pipeline එක save කිරීම. |
| 91 | `model.feature_names_in_` | Scaled NumPy array මත fit කළ RF එකට මේ attribute එක නොතිබිය හැක; helper call එක fail වේ. | DataFrame input සහ saved Pipeline/schema. |
| 94–96 | Numeric-coded input widgets; manually supplied `feature_names` | Region/Depot/techs mappings UI එක දන්නේ නැත; wrong codes valid-looking scores දෙයි. Month example එකේ 12 original 0–11 mapping සමග නොගැළපේ. | Raw category selectors, names/IDs, month/day validation සහ same normalization. |
| 96 | `%%writefile ...` පසුව `@st.cache_resource` | Generated file එකේ `streamlit`, `joblib`, `pd`, `np` ආදී imports නොමැත; standalone launch එක fail වේ. | Complete standalone `app.py`. |
| 97 | `!streamlit run ...` | Notebook Run All එක server process එකේ නතර විය හැක. | Server launch Anaconda Prompt instructions තුළ පමණක්. |
| 98 | `input_path = 'Model2.ipynb'` | Current copy එක `Model2(1).ipynb` නම් වැරදි/වෙනත් file එක export විය හැක. | Corrected notebook export සඳහා hard-coded unrelated notebook path එක ඉවත් කළා. |

## Methodological ගැටලු සහ සීමා

1. **Cells 3, 9, 91, 94–96 — future-failure claim.** `There is a ... probability that the site will go down` යන වාක්‍යයට data එකේ future label/horizon නැත. `repair_time` සහ assigned `techs` incident එකෙන් පසු දැනගන්නා values විය හැක. Future prediction ලෙස එය target-related leakage විය හැක. මේ deliverable එක **recorded site-down status retrospective classification** ලෙස සීමා කර ඇත. Advance-warning model එකක් අවශ්‍ය නම් pre-event inputs සහ future labels අවශ්‍යය; මෙහි ඒ අලුත් analysis එක කර නැත.
2. **Cells 31–34 — selection before CV.** Full dataset target associations මත features drop කර පසුව CV කළොත් test labels selection එකට ඇතුළත් වේ. Original p<0.05 selection rule එක සෑම training fold එකක් තුළට මාරු කර ඇත. `tower_id` group identifier ලෙස වෙන් කර ඇත. `affected_service_count` සහ redundant `Days` original පරිදි inputs නොවේ. Network/vendor ඇතුළු original candidate variables training fold එකේ selection සඳහා පවතී; whole-data significance මත කලින් ඉවත් නොකරයි.
3. **Cells 39–58 — arbitrary category codes.** Depot/Region/technician codes අතර numeric distance අර්ථවත් නොවේ. විශේෂයෙන් LR/ANN සඳහා එය වැරදි ordering එකක් impose කරයි. Training-fold-only OneHotEncoder භාවිත කර ඇත; numeric repair_time සඳහා imputation/scaling. Saved model තුළ mappings පවතී.
4. **Cells 65, 69, 74, 79, 89 — repeated tower rows.** Original random row split එකේ එකම tower train/test දෙකට යා හැක. Same 12-fold `StratifiedGroupKFold` by tower_id දැන් සියලු models සඳහා භාවිත වේ. මේ test එක same synthetic period එකේ unseen towers සඳහාය; future months සඳහා performance තහවුරු නොකරයි. Region/depot/technician overlap තවම තිබිය හැක.
5. **Cell 69 — outer fold as validation_data.** Fixed epochs 5 සහ early stopping නොමැති නිසා එය තමන් විසින් training-label leakage බවට සාක්ෂියක් නොවේ. නමුත් එම curves මත tuning කළොත් test set භාවිත වේ. මෙහි tuning නොකර, fold-loss curves diagnostics ලෙස පමණක් දැක්වේ. Random seeds දමා ඇත; hardware/library වෙනස්වීමෙන් ANN values සුළු ලෙස වෙනස් විය හැක.
6. **Cell 32 — causal comments.** Association එකෙන් causal effect තහවුරු නොවේ. Day-of-month values වලින් weekday/weekend effect කියන්න බැහැ. P-values බහු පරීක්ෂණ සඳහා adjusted නැත; sparse categories සඳහා chi-square approximation සීමා සහිතය. Original exploratory selection rule තබා, causal statements ඉවත් කර ඇත.
7. **Cells 86–89 — “best model” සහ score claims.** Accuracy පමණක් සහ last-fold reports මත best model තෝරන එක ප්‍රමාණවත් නැත. Pooled precision/recall/F1 ඇතුළු metrics දැක්වේ. Balanced RF raw `predict_proba` calibrated real-world probability ලෙස නොපෙන්වයි. Original 0.4/0.7 cutoffs illustrative score bands පමණි.

## නිවැරදි කළ scope එකේ ඉතිරි limitations

මෙම dataset එකෙන් original future forecasting objective එක verify කළ නොහැක. Dashboard එක descriptive/retrospective research සඳහා පමණි. Independent real-world test data නොමැති බව පැහැදිලිව දක්වා ඇත. Historical exploratory feature/model choices මෙම dataset එකම බලා කර ඇති නිසා නව external validation එකක් සෑදෙන්නේ නැත. Model families, ANN epochs/batch size සහ existing estimator hyperparameters tune කර නැත. Encoding සහ split correction නිසා metrics පැරණි notebook එකට සමාන විය යුතු නැත. DT එකේ දැඩි pruning සහ one-hot representation නිසා majority-class predictions වැඩි විය හැක; එය වසන් කර නැත.

Execution සහ dashboard tests පිළිබඳ නිශ්චිත විස්තර `TESTING.md` තුළ ඇත.
