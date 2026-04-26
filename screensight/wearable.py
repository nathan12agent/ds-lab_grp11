"""
screensight/wearable.py
───────────────────────
Pure-Python helpers with no Streamlit dependency:
  - derive_bia_fields()   — auto-derive 14 BIA body-composition fields
  - parse_watch_file()    — parse wearable CSV/XLSX/Parquet (Formats A/B/C)
  - export_report()       — generate printable HTML report string with EDA
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    if _ACTIGRAPHY_KEYS.issubset(cols_set):
        row    = df.iloc[0]
        values = {k: float(row[k]) for k in _ACTIGRAPHY_KEYS}
        values["relative_date_PCIAT_mean"] = 14.0
        return {"values": values, "format": "C", "estimated": False}

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
# EDA CHART HELPERS
# ─────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _load_eda_data() -> pd.DataFrame | None:
    """Try to load the training dataset for EDA charts."""
    try:
        df = pd.read_csv("clean_train_final1 (1).csv")
        df = df.drop(columns=["id", "PCIAT-PCIAT_Total"], errors="ignore")
        df = df.fillna(df.median(numeric_only=True))
        df["sii"] = df["sii"].replace(3, 2)
        return df
    except Exception:
        return None


def _make_eda_charts(answers: dict, final_pred: int) -> dict:
    """
    Generate EDA charts as base64 PNG strings.
    Returns dict of chart_name -> base64 string.
    If dataset not found returns empty dict.
    """
    df = _load_eda_data()
    if df is None:
        return {}

    charts = {}
    BG     = "#0d1226"
    sii_names  = {0: "None", 1: "Mild", 2: "Moderate"}
    sii_colors = ["#00e5a0", "#5c6fff", "#ffb547"]

    user_age    = float(answers.get("Basic_Demos-Age", 0))
    user_screen = float(answers.get("PreInt_EduHx-computerinternet_hoursday", 0))
    user_bmi    = float(answers.get("Physical-BMI", 0))

    # ── Chart 1: SII Distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sii_counts = df["sii"].value_counts().sort_index()
    bars = ax.bar(
        [sii_names.get(i, str(i)) for i in sii_counts.index],
        sii_counts.values,
        color=sii_colors[:len(sii_counts)],
        edgecolor="none", width=0.5,
    )
    user_sii_name = sii_names.get(final_pred, str(final_pred))
    for bar, label in zip(bars, [sii_names.get(i, str(i)) for i in sii_counts.index]):
        if label == user_sii_name:
            bar.set_edgecolor("#ffffff"); bar.set_linewidth(2)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    "← You", ha="center", color="#ffffff", fontsize=8, fontweight="bold")
    ax.set_title("SII Class Distribution", color="#c5cae9", fontsize=9, pad=8)
    ax.set_ylabel("Count", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["sii_dist"] = _fig_to_base64(fig)

    # ── Chart 2: Age Distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.hist(df["Basic_Demos-Age"].dropna(), bins=20, color="#5c6fff", edgecolor="none", alpha=0.8)
    ax.axvline(user_age, color="#ffffff", linewidth=2, linestyle="--")
    ax.text(user_age + 0.3, ax.get_ylim()[1] * 0.85, "You", color="#ffffff", fontsize=8, fontweight="bold")
    ax.set_title("Age Distribution", color="#c5cae9", fontsize=9, pad=8)
    ax.set_xlabel("Age", color="#8892b0", fontsize=8)
    ax.set_ylabel("Count", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["age_dist"] = _fig_to_base64(fig)

    # ── Chart 3: Screen Time Distribution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.hist(df["PreInt_EduHx-computerinternet_hoursday"].dropna(), bins=15, color="#ffb547", edgecolor="none", alpha=0.8)
    ax.axvline(user_screen, color="#ffffff", linewidth=2, linestyle="--")
    ax.text(user_screen + 0.1, ax.get_ylim()[1] * 0.85, "You", color="#ffffff", fontsize=8, fontweight="bold")
    ax.set_title("Screen Time Distribution", color="#c5cae9", fontsize=9, pad=8)
    ax.set_xlabel("Hours/day", color="#8892b0", fontsize=8)
    ax.set_ylabel("Count", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["screen_dist"] = _fig_to_base64(fig)

    # ── Chart 4: BMI Distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.hist(df["Physical-BMI"].dropna(), bins=20, color="#00e5a0", edgecolor="none", alpha=0.8)
    ax.axvline(user_bmi, color="#ffffff", linewidth=2, linestyle="--")
    ax.text(user_bmi + 0.2, ax.get_ylim()[1] * 0.85, "You", color="#ffffff", fontsize=8, fontweight="bold")
    ax.set_title("BMI Distribution", color="#c5cae9", fontsize=9, pad=8)
    ax.set_xlabel("BMI", color="#8892b0", fontsize=8)
    ax.set_ylabel("Count", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["bmi_dist"] = _fig_to_base64(fig)

    # ── Chart 5: Screen Time boxplot by SII class ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sii_groups     = [df[df["sii"] == i]["PreInt_EduHx-computerinternet_hoursday"].dropna().values
                      for i in sorted(df["sii"].unique())]
    sii_labels_box = [sii_names.get(i, str(i)) for i in sorted(df["sii"].unique())]
    bp = ax.boxplot(sii_groups, labels=sii_labels_box, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#8892b0"),
                    capprops=dict(color="#8892b0"),
                    flierprops=dict(marker="o", color="#8892b0", markersize=3))
    for patch, color in zip(bp["boxes"], sii_colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(user_screen, color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(0.5, user_screen + 0.1, f"Your screen time: {user_screen:.0f}h",
            color="#ffffff", fontsize=8)
    ax.set_title("Screen Time by SII Class", color="#c5cae9", fontsize=9, pad=8)
    ax.set_ylabel("Hours/day", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["screen_box"] = _fig_to_base64(fig)

    # ── Chart 6: Percentile bars ──────────────────────────────────────────────
    percentile_data = {
        "Screen Time":    ("PreInt_EduHx-computerinternet_hoursday", user_screen),
        "BMI":            ("Physical-BMI", user_bmi),
        "Age":            ("Basic_Demos-Age", user_age),
        "Sleep Score":    ("SDS-SDS_Total_Raw", float(answers.get("SDS-SDS_Total_Raw", 35))),
        "Activity Score": ("PAQ_C-PAQ_C_Total", float(answers.get("PAQ_C-PAQ_C_Total", 2.5))),
    }
    pct_labels, pct_values, pct_colors = [], [], []
    for label, (feat, val) in percentile_data.items():
        if feat in df.columns:
            pct = float((df[feat].dropna() <= val).mean() * 100)
            pct_labels.append(label)
            pct_values.append(pct)
            pct_colors.append("#ff4d6d" if pct > 75 else "#ffb547" if pct > 50 else "#00e5a0")

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(pct_labels, pct_values, color=pct_colors, height=0.5, edgecolor="none")
    ax.set_xlim(0, 115)
    ax.axvline(50, color=(1,1,1,0.15), linewidth=1, linestyle="--")
    for bar, val, color in zip(bars, pct_values, pct_colors):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}th", va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_title("Your Percentile Rankings vs Dataset", color="#c5cae9", fontsize=9, pad=8)
    ax.set_xlabel("Percentile", color="#8892b0", fontsize=8)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color=(1,1,1,0.06)); ax.set_axisbelow(True)
    charts["percentiles"] = _fig_to_base64(fig)

    return charts


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
    sii_label, _, sii_desc = SII_LABELS.get(result, ("Unknown", "", ""))
    badge_colours = {0: "#00e5a0", 1: "#5c6fff", 2: "#ffb547", 3: "#ff4d6d"}
    badge_colour  = badge_colours.get(result, "#5c6fff")

    # Model prediction rows
    model_rows = ""
    for model_name, pred in predictions.items():
        pred_label  = SII_LABELS.get(pred, (str(pred), "", ""))[0]
        pred_colour = badge_colours.get(pred, "#fff")
        model_rows += (
            f"<tr><td>{model_name}</td>"
            f"<td style='color:{pred_colour};font-weight:600'>"
            f"{pred} \u2014 {pred_label}</td></tr>"
        )

    # Key answer rows
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

    # Generate EDA charts
    charts = _make_eda_charts(answers, result)

    # Build EDA section HTML
    if charts:
        eda_section = f"""
  <div class="card">
    <h2>📊 How You Compare to the Dataset</h2>
    <p class="eda-note">Charts show the full training dataset distribution. The white dashed line marks your position.</p>

    <div class="chart-grid">
      <div class="chart-item">
        <img src="data:image/png;base64,{charts.get('sii_dist', '')}" alt="SII Distribution"/>
      </div>
      <div class="chart-item">
        <img src="data:image/png;base64,{charts.get('age_dist', '')}" alt="Age Distribution"/>
      </div>
      <div class="chart-item">
        <img src="data:image/png;base64,{charts.get('screen_dist', '')}" alt="Screen Time Distribution"/>
      </div>
      <div class="chart-item">
        <img src="data:image/png;base64,{charts.get('bmi_dist', '')}" alt="BMI Distribution"/>
      </div>
    </div>

    <div class="chart-full">
      <img src="data:image/png;base64,{charts.get('screen_box', '')}" alt="Screen Time by SII Class"/>
    </div>

    <div class="chart-full">
      <img src="data:image/png;base64,{charts.get('percentiles', '')}" alt="Percentile Rankings"/>
    </div>
  </div>
"""
    else:
        eda_section = """
  <div class="card">
    <h2>📊 Dataset EDA</h2>
    <p style="color:#8892b0;">Dataset not available for chart generation.</p>
  </div>
"""

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
        padding:40px 24px;max-width:900px;margin:0 auto}}
  h1,h2,h3{{font-family:'Syne',sans-serif;letter-spacing:-0.02em}}
  h1{{font-size:2rem;margin-bottom:4px;color:#fff}}
  .subtitle{{color:#8892b0;font-size:0.9rem;margin-bottom:32px}}
  .card{{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
         border-radius:16px;padding:24px 28px;margin-bottom:24px}}
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
  .eda-note{{color:#8892b0;font-size:0.82rem;margin-bottom:20px}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  .chart-item img{{width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.06)}}
  .chart-full{{margin-bottom:16px}}
  .chart-full img{{width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.06)}}
  .footer{{color:#4a5568;font-size:0.78rem;text-align:center;margin-top:40px}}
  @media print{{
    body{{background:#080c1a !important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  }}
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

  {eda_section}

  <p class="footer">Generated by ScreenSight v2 &mdash; for informational purposes only. Not a clinical diagnosis.</p>
</body>
</html>"""