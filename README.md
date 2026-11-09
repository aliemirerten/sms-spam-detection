# SMS Spam Detection

Simple command-line NLP project by Erdem, Enes, and Emir. The model is a
`CountVectorizer` + `MultinomialNB(alpha=1.0)` scikit-learn pipeline.
`ham` means a legitimate message; `spam` means an unwanted message.

## Setup (Windows PowerShell / VS Code)

Open this folder in VS Code. Use Python 3.12 (tested with 3.12.10).
If `.venv` already exists, skip its creation. Activation is optional; all commands
below call the environment's Python directly, so no execution-policy change is needed.

```powershell
# Create the environment on a new machine with Python on PATH:
python -m venv .venv
# If Python is installed in the standard per-user location but is not on PATH:
# & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe predict.py
```

In VS Code, use **Python: Select Interpreter** and select
`.venv\Scripts\python.exe`.

`requirements.txt` pins the four main dependencies. To reproduce all installed
package versions from this run, use `requirements-lock.txt` instead:
`.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.

Single prediction:

```powershell
.\.venv\Scripts\python.exe predict.py --message "Are we still meeting at six?"
```

Interactive mode accepts English SMS messages and exits with `/quit` or Ctrl+C.
The first training run needs internet to download UCI data. Later training and
prediction runs work offline using the cached data/model. Train again if the model
file is missing. Load only the model produced locally by this project.

## Processing steps

1. `data.py` downloads the official ZIP, keeps its original files, and records
   source URLs, download time, and SHA-256 checksums.
2. Parse each physical record at its first tab; SMS quotes are literal text.
   Audit missing fields, malformed lines, unknown labels, and duplicates.
3. Identify repeated messages using case folding and collapsed whitespace.
   Remove invalid records, remove every record in conflicting-label groups,
   then retain the first occurrence of each remaining message. Original SMS
   text is preserved for modeling. Punctuation is retained; near-duplicate
   templates with different wording/numbers are not deduplicated.
4. Split cleaned data with `test_size=0.20`, `random_state=42`, and
   `stratify=label`. Assert zero shared message keys between train and test.
5. Fit the pipeline only on training messages. CountVectorizer uses its defaults:
   lowercase word counts, tokens of at least two word characters, no stop-word
   removal or stemming. Naive Bayes uses additive smoothing with `alpha=1.0`.
6. Fit `DummyClassifier(strategy="most_frequent")` on training labels. Evaluate
   both models on the same untouched test split. The positive class is `spam`.
   Undefined precision is saved as zero (`zero_division=0`).
7. Save results and a training-only model. There is no parameter search or
   test-set-based tuning. Error examples are for inspection only.

## Files and outputs

| File | Purpose |
| --- | --- |
| `data.py` | Official download, parsing, cleaning, audit |
| `train.py` | Fixed split, pipeline, baseline, evaluation, saved model |
| `predict.py` | Single-message or interactive terminal predictions |
| `verify.py` | Cleaning edge cases and independent checks of saved results |
| `outputs/data_audit.json` | Raw/clean counts, missing/duplicate checks, split counts |
| `outputs/source.json` | Data provenance and checksums |
| `outputs/class_distribution.csv` | Class counts at each stage |
| `outputs/removed_records.csv` | Removed records and reasons |
| `outputs/cleaned_data.csv`, `*_split.csv` | Reproducible data/split inspection |
| `outputs/metrics.csv`, `metrics.json` | Accuracy and spam precision, recall, F1 |
| `outputs/confusion_matrix_*.csv`, `*.png` | One confusion matrix per model |
| `outputs/test_predictions.csv` | Every test prediction, including baseline |
| `outputs/misclassified.csv`, `error_examples.csv` | All errors and up to 10 of each type |
| `outputs/run_config.json` | Versions and fixed experiment settings |
| `outputs/verification.json` | Verification result |
| `outputs/spam_model.joblib` | Saved fitted pipeline, excluded from Git |

Confusion matrix **rows = actual**, **columns = predicted**, class order
**[ham, spam]**: `[[TN, FP], [FN, TP]]`. A false positive marks a legitimate SMS
as spam; a false negative misses a spam SMS.

The dataset is historical English SMS data. Results describe this fixed holdout
only; new messages, other languages, or similar templates can behave differently.

## Verified run

| Stage | Total | Ham | Spam |
| --- | ---: | ---: | ---: |
| Raw | 5574 | 4827 | 747 |
| Clean | 5159 | 4517 | 642 |
| Train | 4127 | 3613 | 514 |
| Test | 1032 | 904 | 128 |

Missing/malformed records: 0. Exact repeated rows beyond the first: 403.
After case/whitespace normalization, 415 repeated messages were removed in total
(including those exact repeats). Conflicting-label groups: 0. Train/test overlap: 0.

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| MultinomialNB | 0.984496 | 0.966667 | 0.906250 | 0.935484 |
| DummyClassifier | 0.875969 | 0.000000 | 0.000000 | 0.000000 |

Naive Bayes matrix: `[[900, 4], [12, 116]]`; baseline matrix:
`[[904, 0], [128, 0]]`. Six verification checks passed. Terminal prediction and
interactive mode were also exercised. These values were produced by running the
code; full precision is in `outputs/metrics.json`.

## Dataset source and citation

- Official page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Official archive: https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip
- Dataset citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*
  [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
- License shown by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Original archive documentation is preserved at `data/raw/readme`.
- UCI lists 5,574 instances. The program counts the downloaded file itself;
  actual raw and cleaned counts are recorded in `outputs/data_audit.json`.

This repository contains the technical implementation and usage instructions,
not academic report prose.
