"""Train once on an 80/20 split, compare a baseline, and save real results."""

import os
import platform
from importlib.metadata import version

import joblib
from data import (LABELS, OUTPUT_DIR, ROOT, class_counts, clean_records, download_data,
                  read_records, save_json)

# Keep plotting caches inside the project, including in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])


def evaluate(name, actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label="spam", zero_division=0
    )
    return {"model": name, "accuracy": float(accuracy_score(actual, predicted)),
            "spam_precision": float(precision), "spam_recall": float(recall),
            "spam_f1": float(f1), "test_records": len(actual)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path, source = download_data()
    raw = read_records(path)
    clean, removed, audit = clean_records(raw)
    train, test = train_test_split(clean, test_size=0.20, random_state=42,
                                   stratify=clean["label"])
    overlap = set(train["message_key"]) & set(test["message_key"])
    assert not overlap, "A message occurs in both train and test."
    audit.update({"source": source, "train_records": len(train), "test_records": len(test),
                  "train_class_counts": class_counts(train),
                  "test_class_counts": class_counts(test),
                  "train_test_message_overlap": len(overlap)})
    clean.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)
    removed.to_csv(OUTPUT_DIR / "removed_records.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    save_json(OUTPUT_DIR / "data_audit.json", audit)
    save_json(OUTPUT_DIR / "source.json", source)
    pd.DataFrame([{"stage": stage, "total": len(frame), **class_counts(frame)}
                  for stage, frame in [("raw", raw), ("clean", clean),
                                       ("train", train), ("test", test)]]).to_csv(
        OUTPUT_DIR / "class_distribution.csv", index=False)

    # Only training messages reach fit(): the test vocabulary is never learned.
    model = build_model()
    model.fit(train["message"], train["label"])
    predicted = model.predict(test["message"])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit([[0]] * len(train), train["label"])
    baseline_predicted = baseline.predict([[0]] * len(test))
    results = [evaluate("MultinomialNB", test["label"], predicted),
               evaluate("DummyClassifier", test["label"], baseline_predicted)]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    matrices = {}
    for result, predictions in zip(results, [predicted, baseline_predicted]):
        name = result["model"]
        matrix = confusion_matrix(test["label"], predictions, labels=LABELS)
        matrices[name] = matrix.tolist()
        pd.DataFrame(matrix, index=["actual_ham", "actual_spam"],
                     columns=["predicted_ham", "predicted_spam"]).to_csv(
            OUTPUT_DIR / f"confusion_matrix_{name}.csv", index_label="actual_class")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set(title=f"{name} - Test Set", xlabel="Predicted label", ylabel="Actual label")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png", dpi=160)
        plt.close(fig)
    save_json(OUTPUT_DIR / "metrics.json", {
        "positive_class": "spam", "zero_division": 0, "models": results,
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "class_order": LABELS},
        "confusion_matrices": matrices,
    })
    predictions = test[["source_line", "label", "message"]].rename(columns={"label": "actual_label"}).copy()
    predictions["predicted_label"] = predicted
    predictions["baseline_prediction"] = baseline_predicted
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    errors = predictions.loc[predictions["actual_label"] != predictions["predicted_label"]].copy()
    errors["error_type"] = errors["actual_label"].map({"ham": "false_positive", "spam": "false_negative"})
    errors.to_csv(OUTPUT_DIR / "misclassified.csv", index=False)
    # Include up to 10 examples of each error type for manual inspection only.
    errors.groupby("error_type", sort=True).head(10).to_csv(
        OUTPUT_DIR / "error_examples.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / "spam_model.joblib")
    save_json(OUTPUT_DIR / "run_config.json", {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]},
        "test_size": 0.2, "random_state": 42, "stratify": "label",
        "vectorizer": "CountVectorizer(default parameters)", "alpha": 1.0,
        "vocabulary_size": len(model.named_steps["vectorizer"].vocabulary_),
        "baseline_strategy": "most_frequent", "baseline_class": str(train["label"].mode().iloc[0]),
        "fit_scope": "training split only", "hyperparameter_search": False,
    })
    print(f"Raw: {audit['raw_records']} | Clean: {len(clean)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Train/test message overlap: {len(overlap)}")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Misclassified messages: {len(errors)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
