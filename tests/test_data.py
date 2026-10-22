import hashlib
import json

import pandas as pd
import pytest

from c1 import LABELS
from c1 import data


def write_raw(path, train=None, test=None):
    path.mkdir(parents=True, exist_ok=True)
    if train is None:
        train = pd.DataFrame([
            {"id": "001", "text": "OlÃ¡ ðŸ‘‹\n<script>naÃ¯ve</script>", "language": "pt", "label": LABELS[0]},
            {"id": "002", "text": "NA", "language": "en", "label": LABELS[1]},
        ])
    if test is None:
        test = pd.DataFrame([{"id": "099", "text": "bonjour", "language": "fr"}])
    train.to_csv(path / "train.csv", index=False)
    test.to_csv(path / "test.csv", index=False)
    pd.DataFrame({"id": test.id, "label": LABELS[0]}).to_csv(path / "sample_submission.csv", index=False)
    return train, test


def test_load_preserves_unicode_newlines_ids_and_literal_na(tmp_path):
    original, _ = write_raw(tmp_path)
    train, test, sample = data.load_competition(tmp_path)
    pd.testing.assert_frame_equal(train, original)
    assert test.id.tolist() == sample.id.tolist() == ["099"]


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "blank", "label", "language", "sample_order"])
def test_rejects_invalid_competition_schema_and_values(tmp_path, kind):
    train, test = write_raw(tmp_path)
    if kind == "missing":
        train = train.drop(columns="text")
    elif kind == "extra":
        train["unexpected"] = "x"
    elif kind == "duplicate":
        train.loc[1, "id"] = "001"
    elif kind == "blank":
        train.loc[0, "text"] = " \n\t"
    elif kind == "label":
        train.loc[0, "label"] = "OTHER"
    elif kind == "language":
        train.loc[0, "language"] = "vi"
    else:
        pd.DataFrame({"id": ["wrong"], "label": [LABELS[0]]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    train.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError):
        data.load_competition(tmp_path)


def test_hashes_detect_tampering(tmp_path):
    write_raw(tmp_path)
    hashes = data.compute_hashes(tmp_path)
    assert hashes["train.csv"] == hashlib.sha256((tmp_path / "train.csv").read_bytes()).hexdigest()
    data.assert_hashes(tmp_path, hashes)
    with (tmp_path / "train.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash"):
        data.assert_hashes(tmp_path, hashes)


def training_fixture():
    return pd.DataFrame([
        {"id": f"{label}-{lang}-{i:03}", "text": f"unique{label}{lang}{i} id{label}{lang}{i} token{label}{lang}{i}", "language": lang, "label": label}
        for label in LABELS for lang in ("en", "pt", "fr", "it") for i in range(24)
    ])


def test_prepare_is_complete_deterministic_and_group_isolated(tmp_path):
    raw = tmp_path / "raw"
    train = training_fixture()
    train.loc[1, "text"] = train.loc[0, "text"]
    write_raw(raw, train)
    before = data.compute_hashes(raw)
    first = data.prepare(raw, tmp_path / "first")
    second = data.prepare(raw, tmp_path / "second")
    a = data.read_manifest(tmp_path / "first")
    b = data.read_manifest(tmp_path / "second")
    pd.testing.assert_frame_equal(a, b)
    assert first == second
    assert a.id.tolist() == train.id.tolist()
    assert a.id.is_unique and len(a) == len(train)
    assert a.groupby("group").split.nunique().max() == 1
    assert a.loc[0, "group"] == a.loc[1, "group"] == min(train.id[:2])
    assert set(a.fold) == set(range(20))
    assert set(a[a.fold.isin([0, 1, 2])].split) == {"audit"}
    assert set(a[a.fold.isin([3, 4, 5])].split) == {"dev"}
    assert set(a[a.fold >= 6].split) == {"train"}
    for split in ("train", "dev", "audit"):
        assert set(train[a.split == split].label) == set(LABELS)
    assert first["hashes"] == before == data.compute_hashes(raw)
    assert first["counts"]["train"] == 288
    assert first["near_duplicates"]["train_train_pairs"] == 1
    assert (tmp_path / "first" / "DATA_REPORT.md").is_file()
    assert json.loads((tmp_path / "first" / "data_report.json").read_text(encoding="utf-8")) == first
    assert len(pd.read_csv(tmp_path / "first" / "near_pairs.csv")) == 1


def test_test_similarity_cannot_bridge_train_groups_and_short_texts_are_preserved(tmp_path):
    train = training_fixture()
    # The test sample is >=.8 similar to each endpoint; endpoints are <.8.
    words = [f"w{i}" for i in range(40)]
    train.loc[0, "text"] = " ".join(["left"] + words)
    train.loc[1, "text"] = " ".join(words + ["right"])
    # For a genuine test-only bridge, increase divergent tails to six tokens.
    train.loc[0, "text"] = " ".join([f"left{i}" for i in range(6)] + words)
    train.loc[1, "text"] = " ".join(words + [f"right{i}" for i in range(6)])
    train.loc[2, "text"] = "ðŸ‘‹"
    train.loc[3, "text"] = "ðŸš€"
    test = pd.DataFrame([{"id": "test-bridge", "text": " ".join(words), "language": "en"}])
    write_raw(tmp_path / "raw", train, test)
    report = data.prepare(tmp_path / "raw", tmp_path / "out")
    manifest = data.read_manifest(tmp_path / "out")
    assert manifest.loc[0, "group"] != manifest.loc[1, "group"]
    assert manifest.loc[2, "group"] != manifest.loc[3, "group"]
    assert report["near_duplicates"]["train_test_pairs"] == 2


@pytest.mark.parametrize("target", ["raw", "raw/out", "."])
def test_prepare_rejects_output_overlapping_raw_directory(tmp_path, target):
    write_raw(tmp_path / "raw")
    with pytest.raises(ValueError, match="overlap"):
        data.prepare(tmp_path / "raw", tmp_path / target)


def test_loader_rejects_malformed_csv_rows_without_shifting_ids(tmp_path):
    write_raw(tmp_path)
    (tmp_path / "train.csv").write_text("id,text,language,label\nextra,001,prompt,en,BENIGN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row"):
        data.load_competition(tmp_path)


def test_manifest_rejects_group_crossing_folds_even_within_one_split(tmp_path):
    pd.DataFrame({"id": ["a", "b"], "group": ["a", "a"], "fold": [6, 7], "split": ["train", "train"]}).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError):
        data.read_manifest(tmp_path)


def test_report_explains_capped_candidates_without_copying_prompt_text(tmp_path):
    train = training_fixture()
    train["text"] = "Private prompt contents never belong in JSON reports"
    write_raw(tmp_path / "raw", train)
    report = data.prepare(tmp_path / "raw", tmp_path / "out")
    assert report["near_duplicates"]["capped_buckets"] == 8
    assert report["near_duplicates"]["candidate_pairs"] == 0
    assert "Private prompt contents" not in json.dumps(report)
    assert report["near_duplicates"]["bucket_cap"] < len(train)


def test_test_bucket_inflation_cannot_change_train_groups_or_manifest(tmp_path):
    train = training_fixture()
    train.loc[1, "text"] = train.loc[0, "text"]
    write_raw(tmp_path / "raw", train)
    data.prepare(tmp_path / "raw", tmp_path / "before")
    before = data.read_manifest(tmp_path / "before")
    many_test = pd.DataFrame([
        {"id": f"test-{i}", "text": train.loc[0, "text"], "language": "en"}
        for i in range(201)
    ])
    write_raw(tmp_path / "raw", train, many_test)
    report = data.prepare(tmp_path / "raw", tmp_path / "after")
    after = data.read_manifest(tmp_path / "after")
    assert report["near_duplicates"]["train_train_pairs"] == 1
    assert after.loc[0, "group"] == after.loc[1, "group"]
    pd.testing.assert_frame_equal(before, after)

