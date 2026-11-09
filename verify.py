"""Check cleaning edge cases and independently verify saved experiment outputs."""

import json
import math
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from data import (LABELS, OUTPUT_DIR, RAW_DIR, class_counts, clean_records,
                  normalize_message, read_records, save_json, sha256)


def read_csv(name):
    return pd.read_csv(OUTPUT_DIR / name, keep_default_na=False)


class CleaningChecks(unittest.TestCase):
    def test_duplicates_conflicts_and_missing_values(self):
        samples = [("ham", "Meet ME", True), ("ham", " meet  me ", True),
                   ("ham", "Conflict", True), ("spam", "CONFLICT", True),
                   ("spam", "Win now", True), ("ham", "   ", True),
                   ("", "missing label", True), ("other", "unknown", True),
                   ("ham", "broken", False)]
        raw = pd.DataFrame(samples, columns=["label", "message", "has_separator"])
        raw.insert(0, "source_line", range(1, len(raw) + 1))
        clean, removed, audit = clean_records(raw)
        self.assertEqual(clean["message"].tolist(), ["Meet ME", "Win now"])
        self.assertEqual(audit["invalid_records_removed"], 4)
        self.assertEqual(audit["conflicting_records_removed"], 2)
        self.assertEqual(audit["duplicate_records_removed_after_conflicts"], 1)
        self.assertEqual(len(removed), 7)


class SavedExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_csv("train_split.csv")
        cls.test = read_csv("test_split.csv")
        cls.clean = read_csv("cleaned_data.csv")
        cls.predictions = read_csv("test_predictions.csv")
        cls.model = joblib.load(OUTPUT_DIR / "spam_model.joblib")

    def test_source_and_audit(self):
        audit = json.loads((OUTPUT_DIR / "data_audit.json").read_text(encoding="utf-8"))
        raw_path = RAW_DIR / "SMSSpamCollection"
        self.assertEqual(sha256(raw_path.read_bytes()), audit["source"]["dataset_sha256"])
        raw = read_records(raw_path)
        clean, removed, fresh_audit = clean_records(raw)
        for key, value in fresh_audit.items():
            self.assertEqual(audit[key], value, key)
        self.assertEqual(len(raw), len(clean) + len(removed))
        pd.testing.assert_frame_equal(clean, self.clean)
        self.assertEqual(class_counts(self.test), audit["test_class_counts"])

    def test_split_is_disjoint_and_reproducible(self):
        self.assertEqual(len(self.test), math.ceil(0.2 * len(self.clean)))
        self.assertEqual(len(self.train) + len(self.test), len(self.clean))
        train_keys = set(self.train["message"].map(normalize_message))
        test_keys = set(self.test["message"].map(normalize_message))
        self.assertFalse(train_keys & test_keys)
        self.assertEqual(len(train_keys), len(self.train))
        self.assertEqual(len(test_keys), len(self.test))
        train, test = train_test_split(self.clean, test_size=0.2, random_state=42,
                                       stratify=self.clean["label"])
        self.assertEqual(train["source_line"].tolist(), self.train["source_line"].tolist())
        self.assertEqual(test["source_line"].tolist(), self.test["source_line"].tolist())

    def test_vectorizer_and_classifier_only_fit_training_data(self):
        vectorizer = self.model.named_steps["vectorizer"]
        analyzer = vectorizer.build_analyzer()
        train_tokens = {token for message in self.train["message"] for token in analyzer(message)}
        test_tokens = {token for message in self.test["message"] for token in analyzer(message)}
        self.assertEqual(set(vectorizer.vocabulary_), train_tokens)
        self.assertGreater(len(test_tokens - train_tokens), 0)
        self.assertTrue(set(vectorizer.vocabulary_).isdisjoint(test_tokens - train_tokens))
        classifier = self.model.named_steps["classifier"]
        self.assertEqual(classifier.alpha, 1.0)
        self.assertEqual(classifier.class_count_.sum(), len(self.train))
        for label, count in zip(classifier.classes_, classifier.class_count_):
            self.assertEqual(count, (self.train["label"] == label).sum())

    def test_saved_predictions_and_manual_metrics(self):
        rows = self.predictions
        self.assertEqual(rows["source_line"].tolist(), self.test["source_line"].tolist())
        self.assertEqual(rows["actual_label"].tolist(), self.test["label"].tolist())
        self.assertEqual(rows["message"].tolist(), self.test["message"].tolist())
        self.assertEqual(self.model.predict(self.test["message"]).tolist(), rows["predicted_label"].tolist())
        majority = self.train["label"].mode().iloc[0]
        self.assertTrue(rows["baseline_prediction"].eq(majority).all())
        metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
        csv_metrics = read_csv("metrics.csv").set_index("model")
        for result, column in zip(metrics["models"], ["predicted_label", "baseline_prediction"]):
            actual, predicted = rows["actual_label"], rows[column]
            matrix = [[int(((actual == a) & (predicted == p)).sum()) for p in LABELS] for a in LABELS]
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            expected = {"accuracy": (tp + tn) / len(rows),
                        "spam_precision": tp / (tp + fp) if tp + fp else 0,
                        "spam_recall": tp / (tp + fn) if tp + fn else 0,
                        "spam_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
            self.assertEqual(matrix, metrics["confusion_matrices"][result["model"]])
            csv_matrix = pd.read_csv(OUTPUT_DIR / f"confusion_matrix_{result['model']}.csv", index_col=0)
            np.testing.assert_array_equal(csv_matrix.to_numpy(), matrix)
            for key, value in expected.items():
                self.assertAlmostEqual(result[key], value)
                self.assertAlmostEqual(csv_metrics.loc[result["model"], key], value)

    def test_error_exports_and_png_files(self):
        errors = read_csv("misclassified.csv")
        expected = self.predictions.loc[self.predictions["actual_label"] != self.predictions["predicted_label"]]
        self.assertEqual(errors["source_line"].tolist(), expected["source_line"].tolist())
        self.assertTrue(set(read_csv("error_examples.csv")["source_line"]).issubset(set(errors["source_line"])))
        for name in ["MultinomialNB", "DummyClassifier"]:
            with Image.open(OUTPUT_DIR / f"confusion_matrix_{name}.png") as picture:
                self.assertEqual(picture.format, "PNG")
                picture.verify()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "verification.json", {
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "checks": ["cleaning edge cases", "source checksum and audit",
                   "disjoint reproducible stratified split", "training-only vocabulary and class counts",
                   "saved predictions and independently calculated metrics", "error exports and PNG files"],
    })
    sys.exit(0 if result.wasSuccessful() else 1)
