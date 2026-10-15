"""The registered read's single-arm path (research.scripts.registered_read)."""

from pathlib import Path

from research.scripts import registered_read as rr


def test_a_single_arm_registration_records_a_skipped_comparison():
    comparison, text = rr.comparison_or_skip(None, None, {"ear_monthly": Path("results/run-x/record.json")})
    assert comparison["arms"] == ["ear_monthly"] and "no pairwise comparison" in comparison["skipped"]
    assert text == comparison["skipped"]
