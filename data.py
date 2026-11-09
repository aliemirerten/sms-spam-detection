"""Download the official UCI archive and audit messages before splitting."""

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
LABELS = ["ham", "spam"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def normalize_message(message):
    """Identity key only: ignore case and collapse whitespace, keep punctuation."""
    return " ".join(message.casefold().split())


def class_counts(frame):
    return {label: int((frame["label"] == label).sum()) for label in LABELS}


def download_data():
    """Cache an unmodified archive and its original dataset/readme members."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "sms_spam_collection.zip"
    source_path = RAW_DIR / "source.json"
    if not archive_path.exists():
        print("Downloading the official UCI SMS Spam Collection...")
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            content = response.read()
        # Validate the archive before caching it; extract only known filenames.
        with ZipFile(io.BytesIO(content)) as archive:
            archive.read("SMSSpamCollection")
            archive.read("readme")
        archive_path.write_bytes(content)
        save_json(source_path, {
            "source_url": SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": sha256(content),
            "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.",
            "license": "CC BY 4.0",
            "uci_page_reported_instances": 5574,
        })
    source = json.loads(source_path.read_text(encoding="utf-8"))
    content = archive_path.read_bytes()
    if sha256(content) != source["archive_sha256"]:
        raise ValueError("Cached archive checksum differs from source.json.")
    with ZipFile(io.BytesIO(content)) as archive:
        for name in ("SMSSpamCollection", "readme"):
            (RAW_DIR / name).write_bytes(archive.read(name))
    dataset_path = RAW_DIR / "SMSSpamCollection"
    source["dataset_sha256"] = sha256(dataset_path.read_bytes())
    return dataset_path, source


def read_records(path):
    # Split at the FIRST tab only. Quotes in SMS text are literal characters.
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            label, separator, message = line.rstrip("\r\n").partition("\t")
            rows.append({"source_line": line_number, "label": label,
                         "message": message, "has_separator": bool(separator)})
    return pd.DataFrame(rows)


def clean_records(raw):
    """Apply fixed data-quality rules without using any model predictions."""
    missing_label = raw["label"].str.strip().eq("")
    missing_message = raw["message"].str.strip().eq("")
    malformed = ~raw["has_separator"]
    unknown_label = ~raw["label"].isin(LABELS) & ~missing_label
    invalid = missing_label | missing_message | malformed | unknown_label
    valid = raw.loc[~invalid].copy()
    valid["message_key"] = valid["message"].map(normalize_message)
    label_counts = valid.groupby("message_key")["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    conflict = valid["message_key"].isin(conflict_keys)
    consistent = valid.loc[~conflict].copy()
    duplicates = consistent.duplicated("message_key", keep="first")
    clean = consistent.loc[~duplicates].drop(columns="has_separator").reset_index(drop=True)

    removed_invalid = raw.loc[invalid].copy()
    removed_invalid["reason"] = "missing_malformed_or_unknown_label"
    removed_conflicts = valid.loc[conflict].copy()
    removed_conflicts["reason"] = "conflicting_labels"
    removed_duplicates = consistent.loc[duplicates].copy()
    removed_duplicates["reason"] = "duplicate_normalized_message"
    removed = pd.concat([removed_invalid, removed_conflicts, removed_duplicates], ignore_index=True)
    removed = removed[["source_line", "label", "message", "reason"]]
    audit = {
        "raw_records": len(raw), "raw_class_counts": class_counts(raw),
        "missing_label_records": int(missing_label.sum()),
        "missing_message_records": int(missing_message.sum()),
        "malformed_records": int(malformed.sum()),
        "unknown_label_records": int(unknown_label.sum()),
        "invalid_records_removed": int(invalid.sum()),
        "exact_duplicate_records_beyond_first": int(valid.duplicated(["label", "message"]).sum()),
        "normalized_duplicate_messages_beyond_first": int(valid.duplicated("message_key").sum()),
        "conflicting_message_groups": len(conflict_keys),
        "conflicting_records_removed": int(conflict.sum()),
        "duplicate_records_removed_after_conflicts": int(duplicates.sum()),
        "clean_records": len(clean), "clean_class_counts": class_counts(clean),
        "removed_records": len(removed),
        "identity_rule": "casefold and collapse whitespace; retain punctuation",
        "cleaning_order": ["remove invalid records", "remove all conflicting-label groups",
                           "keep first occurrence of each normalized message"],
        "duplicate_count_note": "Diagnostic duplicate counts overlap; only the removal-stage counts are additive.",
    }
    assert len(raw) == len(clean) + len(removed)
    assert clean["message_key"].is_unique
    return clean, removed, audit
