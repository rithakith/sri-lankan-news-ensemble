# Sri Lankan News Classification

Two notebooks and a compact dashboard for a completed 10,000-article news-classification experiment. The dashboard uses actual exported results and the frozen **TF-IDF + LinearSVC** classifier; it contains no simulated predictions.

## Run the dashboard

Use Python **3.12 or 3.13**. The original model was saved with scikit-learn 1.6.1; keep that version when loading it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

For Streamlit Community Cloud, select `dashboard/app.py` and Python 3.13. Dependencies are in `dashboard/requirements.txt`.

The three views are **Results**, **Predict an article**, and **About the study**. Results provide a model-comparison chart and one selectable heatmap at a time. Prediction accepts a headline and full article body, applies the saved vectorizer, and decodes the classifier's integer labels using its saved topic order. Decision scores are not calibrated probabilities.

## What is included

- `News_Preprocessing.ipynb`: the existing preprocessing notebook.
- `News_Training.ipynb`: latest saved training notebook downloaded from Drive, including outputs and your edits.
- `dashboard/models/final_classifier.joblib`: original frozen model, including TF-IDF, LinearSVC, topic mapping and selection metadata. Joblib is a pickle-based serialization format; changing its extension to `.pkl` would not change the model. No separate vectorizer or retraining is needed.
- `results/`: selected validation, test and reviewed-audit exports, annotation provenance, frozen configuration, report, and a source/checksum manifest.
- `tests/`: metric reproduction, real model inference and dashboard interaction checks.

Large raw corpora, intermediate model caches, repeated predictions, old notebook variants, screenshots and the previous ANN/hybrid dashboard are omitted. The original experiment can regenerate its detailed artifacts. Local pre-cleanup work was backed up under ignored `tmp/github-cleanup-backup/`.

## Actual results

| Measure | Frozen selected-refined LinearSVC | BART |
|---|---:|---:|
| Accuracy on 300 supplied reviewed labels | 64.0% | 58.0% |
| Macro F1 on those labels | 0.5693 | 0.5802 |
| Final-test agreement with BART (1,941 articles) | 77.13% | Reference |

Validation selected LinearSVC at macro F1 **0.6422**. The selected refinement added **315** articles to **3,156** confident seeds. These are separate measures: BART agreement is not independently verified accuracy, and higher reviewed-label accuracy did not imply higher macro F1 than BART.

**Stages:** initial = seeds only; baseline refined = original refinement rule; selected refined = validation-screened rule; weighted refined = lower weights for added pseudo-labels. Each stage compares logistic regression, random forest, XGBoost, LinearSVC and stacking. The human audit compares BART, initial stacking, selected-refined stacking, and the frozen selected-refined LinearSVC; it does not include every model/stage combination.

**Limits:** audit groups are excluded from revised fits and selection, but earlier development may have exposed some articles and earlier test results were viewed. The supplied audit was described by the user as reviewed; its retained notes include AI-assisted annotation provenance and no second independent annotator was verified. Small category samples, shared BART supervision, uncalibrated scores and source/time shifts limit generalization. The dashboard preserves these distinctions.

## Test

```bash
pip install pytest
python -m pytest tests -q
```

The training environment's original package versions are retained in `results/model_requirements.txt`; dashboard dependencies omit XGBoost because inference uses the selected LinearSVC pipeline.
