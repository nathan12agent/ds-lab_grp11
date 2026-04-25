"""
screensight/wearable.py
───────────────────────
Pure-Python helpers with no Streamlit dependency:
  - derive_bia_fields()   — auto-derive 14 BIA body-composition fields
  - parse_watch_file()    — parse wearable CSV/XLSX/Parquet (Formats A/B/C)
  - export_report()       — generate printable HTML report string
"""

import numpy as np
import pandas as pd

from screensight.constants import ACTIGRAPHY_DEFAULTS, SII_LABELS, FEATURE_COLS


# ─────────────────────────────────────────────
# BIA DERIVATION
# ─────────────────────────────────────────────

def derive_bia_fields(
    height_cm: float,
    weight_kg: float,
    age: int,
    sex: int,
    fat_pct: float | None = None,
    activity_level: int = 3,
    frame_num: int = 2,
) -> dict:
    """
    Auto-derive 14 BIA body-composition fields from basic measurements.

    Args:
        height_cm:      Height in centimetres
        weight_kg:      Weight in kilograms
        age:            Age in years
        sex:            1 = Male, 0 = Female
        fat_pct:        Body fat percentage (estimated from sex if None)
        activity_level: 1–5 activity multiplier index
        frame_num:      Body frame size (1=small, 2=medium, 3=large)

    Returns:
        Dict of BIA-prefixed FEATURE_COLS keys → float values
    """
    h_m = height_cm / 100
    bmi = weight_kg / h_m ** 2

    if fat_pct is None:
        fat_pct = 20.0 if sex == 1 else 26.0

    fat_mass = weight_kg * fat_pct / 100
    ffm  = weight_kg - fat_mass
    tbw  = ffm * 0.732
    ecw  = tbw * 0.40
    icw  = tbw * 0.60
    bmc  = weight_kg * 0.035
    smm  = ffm * 0.54
    ldm  = ffm - bmc
    lst  = ffm - smm
    ffmi = ffm / h_m ** 2
    fmi  = fat_mass / h_m ** 2

    # Harris-Benedict BMR
    if sex == 1:
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    else:
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age

    activity_multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    dee = bmr * activity_multipliers.get(activity_level, 1.55)

    return {
        "BIA-BIA_BMI":       round(bmi,      2),
        "BIA-BIA_BMR":       round(bmr,      2),
        "BIA-BIA_TBW":       round(tbw,      2),
        "BIA-BIA_FFM":       round(ffm,      2),
        "BIA-BIA_Fat":       round(fat_pct,  2),
        "BIA-BIA_SMM":       round(smm,      2),
        "BIA-BIA_BMC":       round(bmc,      2),
        "BIA-BIA_ECW":       round(ecw,      2),
        "BIA-BIA_ICW":       round(icw,      2),
        "BIA-BIA_FFMI":      round(ffmi,     2),
        "BIA-BIA_FMI":       round(fmi,      2),
        "BIA-BIA_LDM":       round(ldm,      2),
        "BIA-BIA_LST":       round(lst,      2),
        "BIA-BIA_DEE":       round(dee,      2),
        "BIA-BIA_Frame_num": frame_num,
    }


# ─────────────────────────────────────────────
# WEARABLE FILE PARSING
# ─────────────────────────────────────────────

_ACTIGRAPHY_KEYS = {
    "X_mean", "X_std", "Y_mean", "Y_std", "Z_mean", "Z_std",
    "enmo_mean", "enmo_std", "anglez_mean", "anglez_std",
    "light_mean", "light_std",
}

_FORMAT_B_DEFAULTS = {
    "X_mean": 0.08,  "X_std": 0.28,
    "Y_mean": -0.04, "Y_std": 0.19,
    "Z_mean": -0.77, "Z_std": 0.24,
    "anglez_mean": -9.5, "anglez_std": 19.2,
    "light_mean": 28.0,  "light_std": 47.0,
}


def _safe_mean(series: pd.Series) -> float:
    if series.isna().all():
        return ACTIGRAPHY_DEFAULTS.get(series.name, 0.0)
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    if series.isna().all():
        return ACTIGRAPHY_DEFAULTS.get(series.name, 0.0)
    return float(series.std(ddof=1)) if len(series.dropna()) > 1 else 0.0


def parse_watch_file(uploaded_file) -> dict:
    """
    Parse a Streamlit UploadedFile (.csv, .xlsx, .parquet) containing wearable data.

    Detects three formats:
      A — Raw accelerometer (timestamp, x, y, z, optional light)
      B — Summary export   (steps, active_minutes)
      C — Pre-computed stats matching FEATURE_COLS actigraphy keys

    Returns:
        {"values": dict, "format": "A"|"B"|"C", "estimated": bool}

    Raises:
        ValueError: plain-English message for unrecognised/empty/corrupt files.
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file appears to be empty.")
    except Exception:
        raise ValueError(
            "Could not read the file. Please check it isn't corrupted and try again."
        )

    if df.empty:
        raise ValueError("The uploaded file appears to be empty.")

    cols_lower = {c.lower(): c for c in df.columns}
    cols_set   = set(df.columns)

    # Format C — pre-computed stats
    if _ACTIGRAPHY_KEYS.issubset(cols_set):
        row    = df.iloc[0]
        values = {k: float(row[k]) for k in _ACTIGRAPHY_KEYS}
        values["relative_date_PCIAT_mean"] = 14.0
        return {"values": values, "format": "C", "estimated": False}

    # Format A — raw accelerometer
    if all(k in cols_lower for k in ("timestamp", "x", "y", "z")):
        x = pd.to_numeric(df[cols_lower["x"]], errors="coerce")
        y = pd.to_numeric(df[cols_lower["y"]], errors="coerce")
        z = pd.to_numeric(df[cols_lower["z"]], errors="coerce")

        magnitude = np.sqrt(x**2 + y**2 + z**2)
        enmo      = np.maximum(magnitude - 1.0, 0.0)
        anglez    = np.arctan2(z, np.sqrt(x**2 + y**2)) * (180.0 / np.pi)

        light = (pd.to_numeric(df[cols_lower["light"]], errors="coerce")
                 if "light" in cols_lower
                 else pd.Series([np.nan] * len(df), name="light"))

        # Assign names for fallback lookup in _safe_mean/_safe_std
        x.name = "X_mean";       y.name = "Y_mean";       z.name = "Z_mean"
        enmo.name = "enmo_mean"; anglez.name = "anglez_mean"; light.name = "light_mean"

        def _named_copy(s, name):
            c = s.copy(); c.name = name; return c

        values = {
            "X_mean":    _safe_mean(x),
            "X_std":     _safe_std(_named_copy(x, "X_std")),
            "Y_mean":    _safe_mean(y),
            "Y_std":     _safe_std(_named_copy(y, "Y_std")),
            "Z_mean":    _safe_mean(z),
            "Z_std":     _safe_std(_named_copy(z, "Z_std")),
            "enmo_mean": _safe_mean(enmo),
            "enmo_std":  _safe_std(_named_copy(enmo, "enmo_std")),
            "anglez_mean": _safe_mean(anglez),
            "anglez_std":  _safe_std(_named_copy(anglez, "anglez_std")),
            "light_mean": (_safe_mean(light) if not light.isna().all()
                           else ACTIGRAPHY_DEFAULTS["light_mean"]),
            "light_std":  (_safe_std(_named_copy(light, "light_std")) if not light.isna().all()
                           else ACTIGRAPHY_DEFAULTS["light_std"]),
            "relative_date_PCIAT_mean": 14.0,
        }
        return {"values": values, "format": "A", "estimated": False}

    # Format B — summary export
    if "steps" in cols_lower or "active_minutes" in cols_lower:
        values = dict(_FORMAT_B_DEFAULTS)
        if "steps" in cols_lower:
            steps = pd.to_numeric(df[cols_lower["steps"]], errors="coerce")
            steps_mean = float(steps.mean()) if not steps.isna().all() else 0.0
            values["enmo_mean"] = steps_mean / 10000.0
            values["enmo_std"]  = 0.03
        else:
            values["enmo_mean"] = ACTIGRAPHY_DEFAULTS["enmo_mean"]
            values["enmo_std"]  = ACTIGRAPHY_DEFAULTS["enmo_std"]
        values["relative_date_PCIAT_mean"] = 14.0
        return {"values": values, "format": "B", "estimated": True}

    raise ValueError(
        "We couldn't recognise this file format. Please upload a CSV/XLSX/Parquet "
        "with accelerometer columns (x, y, z) or summary columns (steps, active_minutes)."
    )


# ─────────────────────────────────────────────
# REPORT EXPORT
# ─────────────────────────────────────────────

_PLAIN_LABELS = {
    "Basic_Demos-Age":                        "Age",
    "Basic_Demos-Sex":                        "Sex",
    "CGAS-CGAS_Score":                        "Day-to-day coping score",
    "PreInt_EduHx-computerinternet_hoursday": "Screen time (hours/day)",
    "Physical-Height":                        "Height (cm)",
    "Physical-Weight":                        "Weight (kg)",
    "Physical-BMI":                           "BMI",
    "Physical-HeartRate":                     "Resting heart rate (bpm)",
    "Physical-Systolic_BP":                   "Systolic blood pressure (mmHg)",
    "Physical-Diastolic_BP":                  "Diastolic blood pressure (mmHg)",
    "PAQ_C-PAQ_C_Total":                      "Physical activity score (child)",
    "PAQ_A-PAQ_A_Total":                      "Physical activity score (adolescent)",
    "BIA-BIA_Activity_Level_num":             "Activity level",
    "Fitness_Endurance-Max_Stage":            "Run endurance stage",
    "Fitness_Endurance-Time_Mins":            "Run endurance (minutes)",
    "Fitness_Endurance-Time_Sec":             "Run endurance (seconds)",
    "FGC-FGC_PU":                             "Push-ups",
    "FGC-FGC_CU":                             "Sit-ups",
    "FGC-FGC_SRL":                            "Flexibility — left (cm)",
    "FGC-FGC_SRR":                            "Flexibility — right (cm)",
    "FGC-FGC_GSD":                            "Grip strength — dominant (kg)",
    "FGC-FGC_GSND":                           "Grip strength — non-dominant (kg)",
    "FGC-FGC_TL":                             "Trunk lift (inches)",
    "SDS-SDS_Total_Raw":                      "Sleep disturbance score (raw)",
    "SDS-SDS_Total_T":                        "Sleep disturbance score (T-score)",
    "X_mean":                                 "Wrist tilt forward-back — avg",
    "X_std":                                  "Wrist tilt forward-back — variability",
    "Y_mean":                                 "Wrist tilt side-to-side — avg",
    "Y_std":                                  "Wrist tilt side-to-side — variability",
    "Z_mean":                                 "Wrist tilt up-down — avg",
    "Z_std":                                  "Wrist tilt up-down — variability",
    "enmo_mean":                              "Movement intensity — avg",
    "enmo_std":                               "Movement intensity — variability",
    "anglez_mean":                            "Wrist angle — avg",
    "anglez_std":                             "Wrist angle — variability",
    "light_mean":                             "Ambient light — avg",
    "light_std":                              "Ambient light — variability",
    "relative_date_PCIAT_mean":               "Days since assessment",
    "BIA-BIA_BMI":                            "BMI (BIA)",
    "BIA-BIA_BMR":                            "Basal metabolic rate (kcal/day)",
    "BIA-BIA_TBW":                            "Total body water (L)",
    "BIA-BIA_FFM":                            "Fat-free mass (kg)",
    "BIA-BIA_Fat":                            "Body fat (%)",
    "BIA-BIA_SMM":                            "Skeletal muscle mass (kg)",
    "BIA-BIA_BMC":                            "Bone mineral content (kg)",
    "BIA-BIA_ECW":                            "Extracellular water (L)",
    "BIA-BIA_ICW":                            "Intracellular water (L)",
    "BIA-BIA_FFMI":                           "Fat-free mass index",
    "BIA-BIA_FMI":                            "Fat mass index",
    "BIA-BIA_LDM":                            "Lean dry mass (kg)",
    "BIA-BIA_LST":                            "Lean soft tissue (kg)",
    "BIA-BIA_DEE":                            "Daily energy expenditure (kcal/day)",
    "BIA-BIA_Frame_num":                      "Body frame size",
}

_KEY_FIELDS = [
    "Basic_Demos-Age", "Basic_Demos-Sex",
    "CGAS-CGAS_Score", "PreInt_EduHx-computerinternet_hoursday",
    "Physical-Height", "Physical-Weight", "Physical-BMI", "Physical-HeartRate",
    "PAQ_C-PAQ_C_Total", "BIA-BIA_Activity_Level_num",
    "SDS-SDS_Total_Raw", "enmo_mean", "anglez_mean",
]


def export_report(answers: dict, result: int, predictions: dict) -> str:
    """
    Generate a self-contained printable HTML report.

    Args:
        answers:     dict keyed by FEATURE_COLS names
        result:      SII integer 0–3
        predictions: {model_name: predicted_sii_int}

    Returns:
        Complete HTML string with inline CSS.
    """
    sii_label, _, sii_desc = SII_LABELS.get(result, ("Unknown", "", ""))
    badge_colours = {0: "#00e5a0", 1: "#5c6fff", 2: "#ffb547", 3: "#ff4d6d"}
    badge_colour  = badge_colours.get(result, "#5c6fff")

    model_rows = ""
    for model_name, pred in predictions.items():
        pred_label  = SII_LABELS.get(pred, (str(pred), "", ""))[0]
        pred_colour = badge_colours.get(pred, "#fff")
        model_rows += (
            f"<tr><td>{model_name}</td>"
            f"<td style='color:{pred_colour};font-weight:600'>"
            f"{pred} \u2014 {pred_label}</td></tr>"
        )

    answer_rows = ""
    for key in _KEY_FIELDS:
        if key not in answers:
            continue
        label = _PLAIN_LABELS.get(key, key)
        val   = answers[key]
        if key == "Basic_Demos-Sex":
            val = "Male" if val == 1 else "Female"
        elif isinstance(val, float):
            val = f"{val:.2f}"
        answer_rows += f"<tr><td>{label}</td><td>{val}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ScreenSight v2 — Assessment Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=DM+Sans:wght@400;500&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#080c1a;color:#e8eaf6;font-family:'DM Sans',sans-serif;
        padding:40px 24px;max-width:860px;margin:0 auto}}
  h1,h2,h3{{font-family:'Syne',sans-serif;letter-spacing:-0.02em}}
  h1{{font-size:2rem;margin-bottom:4px;color:#fff}}
  .subtitle{{color:#8892b0;font-size:0.9rem;margin-bottom:32px}}
  .card{{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
         border-radius:16px;padding:24px 28px;margin-bottom:24px;backdrop-filter:blur(12px)}}
  .sii-badge{{display:inline-block;background:{badge_colour};color:#080c1a;
              font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
              padding:6px 20px;border-radius:999px;margin:8px 0 12px}}
  .sii-number{{font-family:'Syne',sans-serif;font-size:4rem;font-weight:700;
               color:{badge_colour};line-height:1}}
  .sii-desc{{color:#b0bec5;margin-top:8px;font-size:0.95rem}}
  h2{{font-size:1.1rem;color:#c5cae9;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:0.9rem}}
  th{{text-align:left;color:#8892b0;font-weight:500;padding:6px 0;
      border-bottom:1px solid rgba(255,255,255,0.08)}}
  td{{padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);color:#e8eaf6}}
  td:last-child{{text-align:right}}
  .footer{{color:#4a5568;font-size:0.78rem;text-align:center;margin-top:40px}}
</style>
</head>
<body>
  <h1>ScreenSight v2</h1>
  <p class="subtitle">Problematic Internet Use Assessment Report</p>
  <div class="card">
    <h2>Assessment Result</h2>
    <div class="sii-number">{result}</div>
    <div class="sii-badge">{sii_label}</div>
    <p class="sii-desc">{sii_desc}</p>
  </div>
  <div class="card">
    <h2>Model Predictions</h2>
    <table><tr><th>Model</th><th>Prediction</th></tr>{model_rows}</table>
  </div>
  <div class="card">
    <h2>Key Answers</h2>
    <table><tr><th>Question</th><th>Answer</th></tr>{answer_rows}</table>
  </div>
  <p class="footer">Generated by ScreenSight v2 &mdash; for informational purposes only. Not a clinical diagnosis.</p>
</body>
</html>"""
