"""Unit tests for _resolve_output_dir in g1_pose_dataset.__main__."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from g1_pose_dataset.__main__ import _parse_args, _resolve_output_dir

FROZEN = datetime(2026, 5, 7, 0, 4, 10)


def _now() -> datetime:
    return FROZEN


def test_default_no_flags_returns_timestamp_only(tmp_path: Path) -> None:
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="",
        resume=False,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410"


def test_default_with_dataset_name_appends_suffix(tmp_path: Path) -> None:
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=False,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"


def test_resume_picks_lexicographically_largest_match(tmp_path: Path) -> None:
    (tmp_path / "20260101_000000_final").mkdir()
    (tmp_path / "20260507_000410_final").mkdir()
    # A name-suffix mismatch must not be picked even though it is "later".
    (tmp_path / "20270101_000000_other").mkdir()
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=True,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"


def test_resume_empty_name_does_not_match_suffixed_folders(tmp_path: Path) -> None:
    (tmp_path / "20260101_000000_final").mkdir()
    with pytest.raises(SystemExit):
        _resolve_output_dir(
            output_dir=None,
            dataset_name="",
            resume=True,
            base=tmp_path,
            now=_now,
        )


def test_resume_with_no_matches_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _resolve_output_dir(
            output_dir=None,
            dataset_name="",
            resume=True,
            base=tmp_path,
            now=_now,
        )


def test_explicit_output_dir_is_returned_verbatim(tmp_path: Path) -> None:
    explicit = tmp_path / "anywhere" / "i" / "want"
    out = _resolve_output_dir(
        output_dir=explicit,
        dataset_name="final",  # ignored
        resume=True,           # ignored
        base=tmp_path,
        now=_now,
    )
    assert out == explicit


def test_resume_ignores_non_matching_filenames(tmp_path: Path) -> None:
    # The match must be a directory whose name matches the regex.
    # The genuine match is the older one — the file with a later timestamp
    # whose name otherwise matches the regex must be filtered out by is_dir().
    (tmp_path / "20260507_000410_final").mkdir()
    (tmp_path / "not_a_timestamp").mkdir()
    (tmp_path / "20270101_000000_final").write_text("file, not dir")
    (tmp_path / "20260507_000410_final.txt").write_text("ignore me")
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=True,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"


def test_resume_missing_base_raises(tmp_path: Path) -> None:
    # First-time --resume with no base directory at all is a common user
    # mistake; must surface a clear error instead of silently creating one.
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit, match="base directory"):
        _resolve_output_dir(
            output_dir=None,
            dataset_name="",
            resume=True,
            base=missing,
            now=_now,
        )


def test_parse_args_default_output_dir_is_none() -> None:
    args = _parse_args([])
    assert args.output_dir is None
    assert args.dataset_name == ""
    assert args.resume is False


def test_parse_args_dataset_name_and_resume_flags() -> None:
    args = _parse_args(["--dataset-name", "final", "--resume"])
    assert args.dataset_name == "final"
    assert args.resume is True


def test_parse_args_explicit_output_dir() -> None:
    args = _parse_args(["--output-dir", "/tmp/foo"])
    assert args.output_dir == Path("/tmp/foo")
