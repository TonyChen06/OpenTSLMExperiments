"""Graders: parsing edge cases and tolerance bands."""

from __future__ import annotations

from opentslm.ahri.graders import (
    extract_all_numbers,
    extract_first_number,
    grade_classification,
    grade_multi_regression,
    grade_multilabel,
    grade_regression,
    match_label,
)


def test_extract_first_number():
    assert extract_first_number("the answer is 3.5 Hz") == 3.5
    assert extract_first_number("negative: -2.7") == -2.7
    assert extract_first_number("no numbers here") is None


def test_extract_all_numbers():
    assert extract_all_numbers("1 2.5 -3 4") == [1.0, 2.5, -3.0, 4.0]
    assert extract_all_numbers("") == []


def test_match_label_word_boundary():
    # 'low' should not match 'slow'
    assert match_label("the signal is slow", ["low", "medium", "high"]) is None
    assert match_label("frequency is low", ["low", "medium", "high"]) == "low"
    # case insensitive
    assert match_label("HIGH FREQUENCY", ["low", "medium", "high"]) == "high"


def test_grade_classification_pass_and_fail():
    r = grade_classification("low frequency", "low", ["low", "medium", "high"])
    assert r["correct"] is True
    r = grade_classification("medium", "low", ["low", "medium", "high"])
    assert r["correct"] is False


def test_grade_regression_tolerance_bands():
    r = grade_regression("5.2", 5.0, (0.5, 1.0, 2.0))
    assert r["correct"] is True
    assert r["hit_0.5"] is True
    assert abs(r["mae"] - 0.2) < 1e-9

    r = grade_regression("8.0", 5.0, (0.5, 1.0, 2.0))
    assert r["correct"] is False
    assert r["hit_2.0"] is False
    assert r["mae"] == 3.0


def test_grade_regression_parse_failure():
    r = grade_regression("not a number", 5.0, (0.5,))
    assert r["correct"] is False
    assert r["mae"] == float("inf")
    assert r["hit_0.5"] is False


def test_grade_multi_regression():
    r = grade_multi_regression("3.0, 7.0", [3.1, 7.2], (0.5,))
    assert r["correct"] is True
    r = grade_multi_regression("3.0", [3.1, 7.2], (0.5,))
    assert r["correct"] is False  # missing second number


def test_grade_multilabel():
    r = grade_multilabel("oscillation, trend", {"oscillation", "trend"}, ["oscillation", "trend", "transient"])
    assert r["correct"] is True
    r = grade_multilabel("oscillation only", {"oscillation", "trend"}, ["oscillation", "trend", "transient"])
    assert r["correct"] is False  # missing 'trend'
    r = grade_multilabel("", set(), ["oscillation", "trend", "transient"])
    assert r["correct"] is True
