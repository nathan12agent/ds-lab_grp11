"""
screensight/constants.py
────────────────────────
All shared constants: feature column list, actigraphy defaults/labels, SII labels.
Nothing here imports from the rest of the package — safe to import anywhere.
"""

FEATURE_COLS = [
    'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score',
    'Physical-BMI', 'Physical-Height', 'Physical-Weight',
    'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
    'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
    'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU',
    'FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_TL',
    'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
    'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
    'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
    'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW',
    'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total',
    'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T',
    'PreInt_EduHx-computerinternet_hoursday',
    'X_mean', 'X_std', 'Y_mean', 'Y_std', 'Z_mean', 'Z_std',
    'enmo_mean', 'enmo_std', 'anglez_mean', 'anglez_std',
    'light_mean', 'light_std', 'relative_date_PCIAT_mean',
]

ACTIGRAPHY_DEFAULTS = {
    "X_mean": 0.08,
    "X_std": 0.28,
    "Y_mean": -0.04,
    "Y_std": 0.19,
    "Z_mean": -0.77,
    "Z_std": 0.24,
    "enmo_mean": 0.038,
    "enmo_std": 0.055,
    "anglez_mean": -9.5,
    "anglez_std": 19.2,
    "light_mean": 28.0,
    "light_std": 47.0,
    "relative_date_PCIAT_mean": 14.0,
}

ACTIGRAPHY_LABELS = {
    "X_mean": "Wrist tilt (forward-back) — avg",
    "X_std": "Wrist tilt (forward-back) — variability",
    "Y_mean": "Wrist tilt (side-to-side) — avg",
    "Y_std": "Wrist tilt (side-to-side) — variability",
    "Z_mean": "Wrist tilt (up-down) — avg",
    "Z_std": "Wrist tilt (up-down) — variability",
    "enmo_mean": "Movement intensity — avg",
    "enmo_std": "Movement intensity — variability",
    "anglez_mean": "Wrist angle — avg",
    "anglez_std": "Wrist angle — variability",
    "light_mean": "Ambient light — avg",
    "light_std": "Ambient light — variability",
    "relative_date_PCIAT_mean": "Days since assessment",
}

SII_LABELS = {
    0: ("None",     "badge-none",     "No significant signs of problematic internet use"),
    1: ("Mild",     "badge-mild",     "Some signs of problematic internet use — monitor and consider lifestyle adjustments"),
    2: ("Moderate", "badge-moderate", "Moderate problematic internet use — consider professional guidance"),
    3: ("Severe",   "badge-severe",   "Severe problematic internet use — professional support is strongly recommended"),
}

# Approximate test-set accuracy for each model (used in results display)
import os, json

def _load_model_accuracy():
    diag_path = os.path.join(os.path.dirname(__file__), "ensemble_diagnostics.json")
    try:
        with open(diag_path) as f:
            diag = json.load(f)
        # Use per-model CV scores from best_params.json if available
        best_path = os.path.join(os.path.dirname(__file__), "best_params.json")
        with open(best_path) as f:
            best = json.load(f)
        return {
            "Logistic Regression": f"{best['Logistic Regression']['grid_search_f1']*100:.1f}%",
            "SVM":                 f"{best['SVM']['grid_search_f1']*100:.1f}%",
            "Random Forest":       f"{best['Random Forest']['grid_search_f1']*100:.1f}%",
            "Gradient Boosting":   f"{best['Gradient Boosting']['grid_search_f1']*100:.1f}%",
            "XGBoost":             f"{best['XGBoost']['grid_search_f1']*100:.1f}%",
        }
    except Exception:
        return {
            "Logistic Regression": "N/A",
            "SVM":                 "N/A",
            "Random Forest":       "N/A",
            "Gradient Boosting":   "N/A",
            "XGBoost":             "N/A",
        }

MODEL_ACCURACY = _load_model_accuracy()
MODEL_WEIGHTS = {
    "Logistic Regression": 0.64,
    "SVM":                 0.86,
    "Random Forest":       0.89,
    "Gradient Boosting":   0.88,
    "XGBoost":             0.88,
}