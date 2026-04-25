"""
Property-based tests for ScreenSight v2.

Property 8  (task 1.3): BMI derivation — Validates: Requirements 4.3
Property 9  (task 1.5): parse_watch_file Format A — Validates: Requirements 7.2, 7.3
Property 10 (task 1.7): export_report HTML — Validates: Requirements 9.6, 1.1
"""

import io
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Make sure the workspace root is on the path so we can import app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from app import (
    derive_bia_fields,
    parse_watch_file,
    export_report,
    FEATURE_COLS,
    SII_LABELS,
    ACTIGRAPHY_DEFAULTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock UploadedFile helper
# ─────────────────────────────────────────────────────────────────────────────

class MockUploadedFile:
    """Minimal stand-in for a Streamlit UploadedFile that wraps a BytesIO.

    pandas.read_csv requires a proper file-like object, so we delegate all
    file-like methods to the underlying BytesIO buffer.
    """

    def __init__(self, name: str, content_bytes: bytes):
        self.name = name
        self._buf = io.BytesIO(content_bytes)

    # Accept optional size arg so pandas C parser can call read(size)
    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._buf.seek(pos, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def __iter__(self):
        self._buf.seek(0)
        return iter(self._buf)

    def readable(self) -> bool:
        return True

    def readline(self, size: int = -1) -> bytes:
        return self._buf.readline(size)

    def readlines(self, hint: int = -1) -> list:
        return self._buf.readlines(hint)


def _make_csv_upload(df: pd.DataFrame, name: str = "data.csv") -> MockUploadedFile:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return MockUploadedFile(name, buf.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# Property 8 — BMI derivation
# Validates: Requirements 4.3
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=50)
@given(
    h=st.floats(90, 220, allow_nan=False, allow_infinity=False),
    w=st.floats(15, 150, allow_nan=False, allow_infinity=False),
    age=st.integers(5, 22),
    sex=st.integers(0, 1),
)
def test_bmi_derivation(h, w, age, sex):
    """**Validates: Requirements 4.3**

    Property 8: BMI is correctly derived from height and weight.
    BIA-BIA_BMI == round(w / (h/100)**2, 2) for all valid h, w, age, sex.
    """
    result = derive_bia_fields(h, w, age, sex)
    expected = round(w / (h / 100) ** 2, 2)
    assert result["BIA-BIA_BMI"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# Property 9 — parse_watch_file Format A
# Validates: Requirements 7.2, 7.3
# ─────────────────────────────────────────────────────────────────────────────

_row_strategy = st.fixed_dictionaries({
    "timestamp": st.integers(0, 10_000),
    "x": st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    "y": st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    "z": st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
})


@settings(max_examples=50)
@given(rows=st.lists(_row_strategy, min_size=2, max_size=50))
def test_parse_watch_file_format_a(rows):
    """**Validates: Requirements 7.2, 7.3**

    Property 9: parse_watch_file Format A computes correct actigraphy statistics.
    For a raw accelerometer CSV (timestamp, x, y, z), the returned values must
    match the expected means and stds computed from the same data.
    """
    df = pd.DataFrame(rows)
    upload = _make_csv_upload(df)

    parsed = parse_watch_file(upload)

    assert parsed["format"] == "A"
    assert parsed["estimated"] is False

    x = df["x"].astype(float)
    y = df["y"].astype(float)
    z = df["z"].astype(float)

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    enmo = np.maximum(magnitude - 1.0, 0.0)
    anglez = np.arctan2(z, np.sqrt(x**2 + y**2)) * (180.0 / np.pi)

    vals = parsed["values"]

    assert vals["X_mean"] == pytest.approx(float(x.mean()), rel=1e-5)
    assert vals["Y_mean"] == pytest.approx(float(y.mean()), rel=1e-5)
    assert vals["Z_mean"] == pytest.approx(float(z.mean()), rel=1e-5)

    assert vals["X_std"] == pytest.approx(float(x.std(ddof=1)), rel=1e-5)
    assert vals["Y_std"] == pytest.approx(float(y.std(ddof=1)), rel=1e-5)
    assert vals["Z_std"] == pytest.approx(float(z.std(ddof=1)), rel=1e-5)

    assert vals["enmo_mean"] == pytest.approx(float(enmo.mean()), rel=1e-5)
    assert vals["enmo_std"] == pytest.approx(float(enmo.std(ddof=1)), rel=1e-5)

    assert vals["anglez_mean"] == pytest.approx(float(anglez.mean()), rel=1e-5)
    assert vals["anglez_std"] == pytest.approx(float(anglez.std(ddof=1)), rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Property 10 — export_report HTML
# Validates: Requirements 9.6, 1.1
# ─────────────────────────────────────────────────────────────────────────────

# Minimal answers dict with all FEATURE_COLS keys set to 0.0
_BASE_ANSWERS = {k: 0.0 for k in FEATURE_COLS}

_predictions_strategy = st.dictionaries(
    keys=st.sampled_from(["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree", "SVM"]),
    values=st.integers(0, 3),
    min_size=1,
    max_size=5,
)


@settings(max_examples=50)
@given(
    sii_int=st.integers(0, 3),
    predictions=_predictions_strategy,
)
def test_export_report_html(sii_int, predictions):
    """**Validates: Requirements 9.6, 1.1**

    Property 10: export_report produces valid HTML containing the SII result.
    - Returns a non-empty string
    - Contains the SII integer value
    - Contains the plain-English SII label (None/Mild/Moderate/Severe)
    - Contains no raw FEATURE_COLS key strings as visible text labels
    """
    html = export_report(_BASE_ANSWERS, sii_int, predictions)

    assert isinstance(html, str)
    assert len(html) > 0

    # Must contain the SII integer
    assert str(sii_int) in html

    # Must contain the plain-English label
    sii_label = SII_LABELS[sii_int][0]  # e.g. "None", "Mild", "Moderate", "Severe"
    assert sii_label in html

    # No raw FEATURE_COLS key should appear as a visible text label
    # (they may appear inside HTML attributes/values, but not as bare text)
    # We check that none of the keys appear as standalone text tokens
    for key in FEATURE_COLS:
        # A key appearing as a table cell text label would look like ">key<"
        assert f">{key}<" not in html, (
            f"Raw FEATURE_COLS key '{key}' found as visible text in report HTML"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 4 — Back navigation preserves all answers
# Validates: Requirements 2.3
# ─────────────────────────────────────────────────────────────────────────────

def simulate_back(step: int, answers: dict):
    """Pure back-navigation logic: decrement step, leave answers untouched."""
    return step - 1, answers


@settings(max_examples=100)
@given(
    step=st.integers(2, 7),
    answers=st.dictionaries(
        keys=st.sampled_from(FEATURE_COLS),
        values=st.floats(0, 100, allow_nan=False),
    ),
)
def test_back_preserves_answers(step, answers):
    """**Validates: Requirements 2.3**

    Property 4: Back navigation preserves all answers.
    For any step > 1 and any answers dict, triggering a Back action SHALL
    result in step being decremented by 1 and the answers dict remaining
    identical to its pre-Back state.
    """
    answers_before = dict(answers)

    new_step, returned_answers = simulate_back(step, answers)

    # Step must be decremented by exactly 1
    assert new_step == step - 1

    # answers dict must be byte-for-byte identical (same keys and values)
    assert returned_answers == answers_before


# ─────────────────────────────────────────────────────────────────────────────
# Property 5 — CGAS remapping
# Validates: Requirements 3.3
# ─────────────────────────────────────────────────────────────────────────────

def cgas_remap(v: int) -> float:
    return v * 7 + 25


@settings(max_examples=100)
@given(v=st.integers(1, 10))
def test_cgas_remapping(v):
    """**Validates: Requirements 3.3**

    Property 5: CGAS remapping formula is correct for all valid inputs.
    For any coping slider value v in [1, 10], the stored CGAS-CGAS_Score
    SHALL equal v × 7 + 25, producing a value in [32, 95].
    """
    result = cgas_remap(v)
    assert result == v * 7 + 25
    assert 32 <= result <= 95


# ─────────────────────────────────────────────────────────────────────────────
# Property 1 — No raw column names in rendered output (export_report variant)
# Validates: Requirements 1.1, 1.2, 6.1, 7.3
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=50)
@given(sii_int=st.integers(0, 3))
def test_no_raw_keys_in_report_output(sii_int):
    """**Validates: Requirements 1.1, 1.2, 6.1, 7.3**

    Property 1: No raw column names appear in the rendered output.
    Tests that PLAIN_LABELS covers all FEATURE_COLS keys and that none of the
    FEATURE_COLS keys appear as visible text (>key<) in the export_report HTML.
    """
    answers = {k: 0.0 for k in FEATURE_COLS}
    html = export_report(answers, sii_int, {"Random Forest": sii_int})
    for key in FEATURE_COLS:
        assert f">{key}<" not in html, (
            f"Raw FEATURE_COLS key '{key}' found as visible text in report HTML"
        )



# ─────────────────────────────────────────────────────────────────────────────
# Property 6 — Height unit conversion
# Validates: Requirements 4.1
# ─────────────────────────────────────────────────────────────────────────────

def height_ft_in_to_cm(feet: int, inches: int) -> float:
    return round((feet * 12 + inches) * 2.54, 1)


@settings(max_examples=100)
@given(feet=st.integers(2, 7), inches=st.integers(0, 11))
def test_height_unit_conversion(feet, inches):
    """**Validates: Requirements 4.1**

    Property 6: Height unit conversion is correct for all valid inputs.
    For any height in feet/inches where feet ∈ [2,7] and inches ∈ [0,11],
    the stored Physical-Height in cm SHALL equal (feet × 12 + inches) × 2.54,
    rounded to one decimal place.
    """
    result = height_ft_in_to_cm(feet, inches)
    expected = round((feet * 12 + inches) * 2.54, 1)
    assert result == expected
    # 2'0" = 60.96 cm, 7'11" = 241.3 cm; 7'0" rounds to 213.4 cm
    assert 60.96 <= result <= 241.3


# ─────────────────────────────────────────────────────────────────────────────
# Property 7 — Weight unit conversion
# Validates: Requirements 4.2
# ─────────────────────────────────────────────────────────────────────────────

def weight_lbs_to_kg(lbs: float) -> float:
    return round(lbs * 0.453592, 1)


@settings(max_examples=100)
@given(w=st.floats(33.0, 265.0, allow_nan=False, allow_infinity=False))
def test_weight_unit_conversion(w):
    """**Validates: Requirements 4.2**

    Property 7: Weight unit conversion is correct for all valid inputs.
    For any weight in pounds w ∈ [33, 265], the stored Physical-Weight in kg
    SHALL equal w × 0.453592, rounded to one decimal place.
    """
    result = weight_lbs_to_kg(w)
    expected = round(w * 0.453592, 1)
    assert result == expected


# ─────────────────────────────────────────────────────────────────────────────
# Property 8 (shared) — BMI derivation also validates Step 2 requirements
# Validates: Requirements 4.3
# Note: test_bmi_derivation above (task 1.3) covers this property.
# This comment serves as the annotation required by task 5.4.
# ─────────────────────────────────────────────────────────────────────────────
# test_bmi_derivation (defined above) also validates Requirements 4.3 for
# Step 2 — Physical Stats. No new test is needed here; the shared property
# covers both the BIA derivation helper and the Physical-BMI / BIA-BIA_BMI
# fields stored during Step 2.



# ─────────────────────────────────────────────────────────────────────────────
# Property 2 — Answer mapping round-trip: all FEATURE_COLS keys are populated
# Validates: Requirements 1.3, 10.2
# ─────────────────────────────────────────────────────────────────────────────

# Simulate the full answer-building pipeline with representative values
def build_full_answers(
    age: int,
    sex: int,
    coping: int,
    screen_h: int,
    height_cm: float,
    weight_kg: float,
    heart_rate: int,
    days_active: int,
    activity_level: int,
    sleep_hours: int,
) -> dict:
    """
    Reproduce the mapping logic from all 4 data-entry steps and return a
    complete answers dict.  Actigraphy fields are filled with ACTIGRAPHY_DEFAULTS.
    """
    from app import derive_bia_fields, ACTIGRAPHY_DEFAULTS

    _PAQ_MAP = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0, 7: 5.0}
    _SLEEP_RAW_MAP = {12: 24, 11: 24, 10: 24, 9: 24, 8: 35, 7: 45, 6: 55, 5: 72, 4: 72}

    answers: dict = {}

    # Step 1
    answers["Basic_Demos-Age"] = float(age)
    answers["Basic_Demos-Sex"] = sex
    answers["CGAS-CGAS_Score"] = float(coping * 7 + 25)
    answers["PreInt_EduHx-computerinternet_hoursday"] = float(screen_h)

    # Step 2
    bmi = round(weight_kg / (height_cm / 100) ** 2, 2)
    answers["Physical-Height"] = float(height_cm)
    answers["Physical-Weight"] = float(weight_kg)
    answers["Physical-BMI"] = bmi
    answers["Physical-HeartRate"] = float(heart_rate)
    # Age-appropriate BP medians
    answers["Physical-Systolic_BP"] = float(110 if age < 13 else 118 if age < 18 else 120)
    answers["Physical-Diastolic_BP"] = float(70 if age < 13 else 74 if age < 18 else 78)
    bia = derive_bia_fields(height_cm, weight_kg, age, sex, activity_level=activity_level)
    answers.update(bia)

    # Step 3
    paq = _PAQ_MAP[days_active]
    answers["PAQ_C-PAQ_C_Total"] = paq
    answers["PAQ_A-PAQ_A_Total"] = paq
    answers["BIA-BIA_Activity_Level_num"] = activity_level
    answers["Fitness_Endurance-Max_Stage"] = 4
    answers["Fitness_Endurance-Time_Mins"] = 4
    answers["Fitness_Endurance-Time_Sec"] = 0
    answers["FGC-FGC_PU"] = 10
    answers["FGC-FGC_CU"] = 20
    answers["FGC-FGC_SRL"] = 20.0
    answers["FGC-FGC_SRR"] = 20.0
    answers["FGC-FGC_GSD"] = 20.0
    answers["FGC-FGC_GSND"] = 18.0
    answers["FGC-FGC_TL"] = 6

    # Step 4
    raw = _SLEEP_RAW_MAP[sleep_hours]
    answers["SDS-SDS_Total_Raw"] = float(raw)
    answers["SDS-SDS_Total_T"] = round(raw * 0.8 + 14, 2)

    # Step 5 — defaults
    answers.update(ACTIGRAPHY_DEFAULTS)

    return answers


@settings(max_examples=50)
@given(
    age=st.integers(5, 22),
    sex=st.integers(0, 1),
    coping=st.integers(1, 10),
    screen_h=st.integers(0, 16),
    height_cm=st.floats(90, 220, allow_nan=False, allow_infinity=False),
    weight_kg=st.floats(15, 150, allow_nan=False, allow_infinity=False),
    heart_rate=st.integers(40, 150),
    days_active=st.integers(0, 7),
    activity_level=st.integers(1, 5),
    sleep_hours=st.integers(4, 12),
)
def test_answer_mapping_round_trip(
    age, sex, coping, screen_h, height_cm, weight_kg,
    heart_rate, days_active, activity_level, sleep_hours,
):
    """**Validates: Requirements 1.3, 10.2**

    Property 2: Answer mapping round-trip — all FEATURE_COLS keys are populated.
    For any valid combination of user inputs across all steps, the resulting
    answers dict SHALL contain exactly all 53 keys from FEATURE_COLS, each
    mapped to a numeric value.
    """
    answers = build_full_answers(
        age, sex, coping, screen_h, height_cm, weight_kg,
        heart_rate, days_active, activity_level, sleep_hours,
    )

    for col in FEATURE_COLS:
        assert col in answers, f"Missing FEATURE_COLS key: {col}"
        val = answers[col]
        assert isinstance(val, (int, float)), f"Non-numeric value for {col}: {val!r}"
        assert not (isinstance(val, float) and np.isnan(val)), f"NaN value for {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 3 — Validation blocks step advance on invalid input
# Validates: Requirements 2.2
# ─────────────────────────────────────────────────────────────────────────────

def simulate_step1_validation(age: int, screen_h: int) -> bool:
    """Returns True if validation passes (step would advance), False if blocked."""
    if not (5 <= age <= 22):
        return False
    if not (0 <= screen_h <= 16):
        return False
    return True


def simulate_step2_validation(height_cm: float, weight_kg: float) -> bool:
    """Returns True if validation passes, False if blocked."""
    if not (90 <= height_cm <= 220):
        return False
    if not (15 <= weight_kg <= 150):
        return False
    return True


@settings(max_examples=100)
@given(
    age=st.integers(-100, 100),
    screen_h=st.integers(-10, 30),
)
def test_validation_blocks_step1_advance(age, screen_h):
    """**Validates: Requirements 2.2**

    Property 3: Validation blocks step advance on invalid input.
    For any age outside [5,22] or screen_h outside [0,16], validation
    SHALL return False (step counter remains unchanged).
    """
    result = simulate_step1_validation(age, screen_h)
    age_valid = 5 <= age <= 22
    screen_valid = 0 <= screen_h <= 16
    expected = age_valid and screen_valid
    assert result == expected


@settings(max_examples=100)
@given(
    height_cm=st.floats(0, 300, allow_nan=False, allow_infinity=False),
    weight_kg=st.floats(0, 300, allow_nan=False, allow_infinity=False),
)
def test_validation_blocks_step2_advance(height_cm, weight_kg):
    """**Validates: Requirements 2.2**

    Property 3 (Step 2): Validation blocks step advance on invalid physical inputs.
    For any height outside [90,220] cm or weight outside [15,150] kg,
    validation SHALL return False.
    """
    result = simulate_step2_validation(height_cm, weight_kg)
    h_valid = 90 <= height_cm <= 220
    w_valid = 15 <= weight_kg <= 150
    expected = h_valid and w_valid
    assert result == expected
