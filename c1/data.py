"""Validation, immutable input fingerprints, and deterministic grouped splits.

Normalization is exclusively for duplicate diagnostics; model text is untouched.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from c1 import LABELS

SCHEMAS = {
    "train.csv": ["id", "text", "language", "label"],
    "test.csv": ["id", "text", "language"],
    "sample_submission.csv": ["id", "label"],
}
LANGUAGES = ("en", "pt", "fr", "it")
BUCKET_CAP = 200


def load_competition(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read strict UTF-8 CSV schemas without interpreting strings such as NA."""
    frames = []
    for filename, columns in SCHEMAS.items():
        path = Path(data_dir) / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != columns:
                raise ValueError(f"{filename}: expected columns {columns}")
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise ValueError(f"{filename}: malformed row {row_number}")
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_filter=False)
        if not len(frame):
            raise ValueError(f"{filename}: empty dataset")
        for column in columns:
            bad = frame[column].isna() | frame[column].str.strip().eq("")
            if bad.any():
                raise ValueError(f"{filename}: blank {column}; row ids {frame.loc[bad, 'id'].tolist()[:10]}")
        if frame.id.duplicated().any():
            raise ValueError(f"{filename}: duplicate ids")
        for column, allowed in (("label", LABELS), ("language", LANGUAGES)):
            if column in frame and not frame[column].isin(allowed).all():
                raise ValueError(f"{filename}: invalid {column}")
        frames.append(frame)
    if frames[1].id.tolist() != frames[2].id.tolist():
        raise ValueError("sample_submission.csv: ids must match test order exactly")
    return tuple(frames)


def compute_hashes(data_dir: str | Path) -> dict[str, str]:
    hashes = {}
    for filename in SCHEMAS:
        digest = hashlib.sha256()
        with (Path(data_dir) / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[filename] = digest.hexdigest()
    return hashes


def assert_hashes(data_dir: str | Path, hashes: dict[str, str]) -> None:
    if compute_hashes(data_dir) != hashes:
        raise ValueError("Raw data hash mismatch: input CSV files changed")


def _shingles(text):
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if len(words) >= 3:
        return {"w:" + " ".join(words[i:i + 3]) for i in range(len(words) - 2)}
    if words:
        return {"short:" + " ".join(words)}
    # Empty word sets must not make unrelated emoji/punctuation compare equal.
    return {"nonword:" + " ".join(text.casefold().split())}


def _near_duplicates(train, test, seed):
    texts = train.text.tolist() + test.text.tolist()
    shingles = [_shingles(text) for text in texts]
    rng = np.random.default_rng(seed)
    prime = np.uint64(4294967291)
    multipliers = rng.integers(1, int(prime), size=24, dtype=np.uint64)
    offsets = rng.integers(0, int(prime), size=24, dtype=np.uint64)
    buckets = defaultdict(list)
    for index, tokens in enumerate(shingles):
        values = np.array([int.from_bytes(hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest(), "little") % int(prime) for token in sorted(tokens)], dtype=np.uint64)
        signature = ((values[:, None] * multipliers + offsets) % prime).min(axis=0)
        for band in range(8):
            buckets[(band, *map(int, signature[band * 3:band * 3 + 3]))].append(index)
    candidates = set()
    capped = 0
    capped_train = 0
    capped_diagnostics = 0
    skipped_entries = 0
    ntrain = len(train)
    for indices in buckets.values():
        training = [index for index in indices if index < ntrain]
        testing = [index for index in indices if index >= ntrain]
        # Test size/content must never affect which train-train edges exist.
        # Apply training caps to training counts before considering diagnostics.
        if len(training) > BUCKET_CAP:
            capped_train += 1
        else:
            for offset, left in enumerate(training):
                for right in training[offset + 1:]:
                    candidates.add((left, right))
        if len(indices) > BUCKET_CAP:
            capped += 1
            skipped_entries += len(indices)
            if training and testing:
                capped_diagnostics += 1
            continue
        for left in training:
            for right in testing:
                candidates.add((left, right))
    parent = list(range(ntrain))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    train_pair_count = 0
    train_nodes = set()
    test_nodes = set()
    ids = train.id.tolist() + test.id.tolist()
    for left, right in sorted(candidates):
        a, b = shingles[left], shingles[right]
        similarity = len(a & b) / len(a | b)
        if similarity < 0.8:
            continue
        kind = "train-train" if right < ntrain else "train-test"
        pairs.append({"left_id": ids[left], "right_id": ids[right], "kind": kind, "jaccard": similarity})
        if right < ntrain:
            parent[find(right)] = find(left)
            train_pair_count += 1
            train_nodes.update((left, right))
        else:
            test_nodes.add(right)
    members = defaultdict(list)
    for i in range(ntrain):
        members[find(i)].append(ids[i])
    names = {root: min(values) for root, values in members.items()}
    groups = [names[find(i)] for i in range(ntrain)]
    stats = {
        "method": "seeded affine MinHash over BLAKE2s-32 word trigrams, 24 permutations, 8 bands of 3; exact Jaccard verification >=0.8",
        "threshold": 0.8, "permutations": 24, "bands": 8, "rows_per_band": 3,
        "bucket_cap": BUCKET_CAP,
        "cap_policy": "train-train candidates use train-only bucket counts; train-test diagnostics separately skip buckets whose combined counts exceed cap",
        "capped_buckets": capped, "skipped_bucket_entries": skipped_entries,
        "capped_train_buckets": capped_train,
        "capped_train_test_buckets": capped_diagnostics,
        "candidate_pairs": len(candidates), "train_train_pairs": train_pair_count,
        "train_test_pairs": len(pairs) - train_pair_count,
        "train_rows_in_groups": len(train_nodes),
        "non_singleton_train_groups": sum(len(values) > 1 for values in members.values()),
        "test_rows_with_train_similarity": len(test_nodes),
        "short_text_method": "one/two words use a single whole-word shingle; zero-word strings use normalized full text, preserving emoji and punctuation",
        "limitations": "Approximate candidates can miss near pairs, especially capped buckets, translations and paraphrases. Case folding and dropped punctuation/emoji in word-bearing strings can inflate similarity. Only train-train edges define groups; no label transfer to test.",
    }
    return groups, pd.DataFrame(pairs, columns=["left_id", "right_id", "kind", "jaccard"]), stats


def _length_stats(frame):
    lengths = frame.text.str.len()
    def summarize(values):
        return {"count": int(len(values)), "min": int(values.min()), "median": float(values.median()), "p95": float(values.quantile(.95)), "max": int(values.max())}
    result = {"all": summarize(lengths), "by_language": {language: summarize(lengths[frame.language == language]) for language in LANGUAGES if (frame.language == language).any()}}
    if "label" in frame:
        result["by_label"] = {label: summarize(lengths[frame.label == label]) for label in LABELS if (frame.label == label).any()}
    return result


def _atomic_write(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(data_dir: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    raw, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Output directory must not overlap the raw data directory")
    hashes = compute_hashes(raw)
    train, test, sample = load_competition(raw)
    groups, pairs, near_stats = _near_duplicates(train, test, seed)
    strata = train.label + "|" + train.language
    if strata.value_counts().min() < 20:
        raise ValueError("20-fold split requires at least 20 rows in each observed label-language stratum")
    folds = np.full(len(train), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(train, strata, groups)):
        folds[indices] = fold
    manifest = pd.DataFrame({"id": train.id, "group": groups, "fold": folds})
    manifest["split"] = np.where(folds < 3, "audit", np.where(folds < 6, "dev", "train"))
    if (folds < 0).any() or manifest.groupby("group").split.nunique().max() != 1:
        raise ValueError("Incomplete or leaking split")
    split_counts = {}
    for split in ("train", "dev", "audit"):
        part = train[manifest.split == split]
        if set(part.label) != set(LABELS):
            raise ValueError(f"Split {split} is missing a label; review large duplicate groups")
        split_counts[split] = {"total": len(part), "by_label_language": {f"{label}|{language}": int(((part.label == label) & (part.language == language)).sum()) for label in LABELS for language in LANGUAGES}}
    marker = re.compile(r"target[\s_\-]*language|langue cible|lingua di destinazione", re.IGNORECASE)
    report = {
        "seed": seed, "hashes": hashes,
        "counts": {"train": len(train), "test": len(test), "sample_submission": len(sample)},
        "language_counts": {name: {str(key): int(value) for key, value in frame.language.value_counts().sort_index().items()} for name, frame in (("train", train), ("test", test))},
        "label_counts": {str(key): int(value) for key, value in train.label.value_counts().items()},
        "length_characters": {"train": _length_stats(train), "test": _length_stats(test)},
        "template_markers": {"pattern": marker.pattern, "train_by_label": {label: int(train.loc[train.label == label, "text"].str.contains(marker).sum()) for label in LABELS}, "test": int(test.text.str.contains(marker).sum())},
        "split_method": "StratifiedGroupKFold 20 folds, label x language; audit 0-2, dev 3-5, train 6-19",
        "split_counts": split_counts, "near_duplicates": near_stats,
    }
    assert_hashes(raw, hashes)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / "manifest.csv", manifest.to_csv(index=False))
    _atomic_write(output / "near_pairs.csv", pairs.to_csv(index=False))
    _atomic_write(output / "data_hashes.json", json.dumps(hashes, indent=2) + "\n")
    _atomic_write(output / "data_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown = "# Data preparation report\n\nRaw CSV text is read unchanged; normalization is used only for similarity diagnostics.\n\n"
    markdown += f"Seed: {seed}. {report['split_method']}.\n\n"
    markdown += "| Split | Rows |\n|---|---:|\n" + "".join(f"| {name} | {values['total']} |\n" for name, values in split_counts.items())
    markdown += "\n## Near duplicates\n\n" + near_stats["method"] + ".\n\n" + near_stats["short_text_method"] + ".\n\n" + near_stats["limitations"] + "\n\n"
    markdown += f"Bucket cap: {BUCKET_CAP}. {near_stats['cap_policy']}. Capped train-only buckets: {near_stats['capped_train_buckets']}; capped train-test diagnostic buckets: {near_stats['capped_train_test_buckets']}. Train-train pairs: {near_stats['train_train_pairs']}; train-test pairs: {near_stats['train_test_pairs']}. Counts are recomputed, not assumed to reproduce an earlier exploratory estimate.\n\n"
    markdown += "All counts, label-language strata, character lengths, template markers, and SHA-256 fingerprints are in `data_report.json`. Prompt text is excluded from reports.\n"
    _atomic_write(output / "DATA_REPORT.md", markdown)
    return report


def read_manifest(prepared_dir: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(Path(prepared_dir) / "manifest.csv", dtype={"id": str, "group": str}, keep_default_na=False)
    if list(manifest) != ["id", "group", "fold", "split"] or not len(manifest) or not manifest.id.is_unique:
        raise ValueError("Invalid split manifest schema or ids")
    if not manifest.fold.isin(range(20)).all() or not manifest.split.isin(("train", "dev", "audit")).all():
        raise ValueError("Invalid manifest fold or split")
    expected = np.where(manifest.fold < 3, "audit", np.where(manifest.fold < 6, "dev", "train"))
    if (manifest.split != expected).any() or manifest.groupby("group").fold.nunique().max() != 1:
        raise ValueError("Leaking or inconsistent manifest")
    return manifest
