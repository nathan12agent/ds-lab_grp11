"""
screensight/ui/steps.py
-----------------------
Render functions for assessment steps 1-6 and the home page.
"""
import numpy as np
import streamlit as st

from screensight.constants import ACTIGRAPHY_DEFAULTS, ACTIGRAPHY_LABELS
from screensight.wearable import derive_bia_fields, parse_watch_file
from screensight.ui.navigation import render_progress_bar

_PAQ_MAP = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0, 7: 5.0}
_ENDURANCE_MAP = {0: (2,2,0), 1: (4,4,0), 2: (7,7,0), 3: (10,10,0), 4: (14,15,0)}
_FLEX_MAP = [8, 20, 30, 38]
_GRIP_MAP = [12, 20, 28, 36]
_SLEEP_RAW_MAP = {12:24, 11:24, 10:24, 9:24, 8:35, 7:45, 6:55, 5:72, 4:72}
_SLEEP_QUALITY_OFFSETS = [0, 5, 10, 15]


def render_home():
    st.markdown("""
<div style="text-align:center;padding:48px 0 32px;">
  <div style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:#fff;line-height:1.15;margin-bottom:12px;">
    Screen<span style="color:#5c6fff;">Sight</span> <span style="font-size:1.8rem;">v2</span>
  </div>
  <div style="font-size:1.1rem;color:#8892b0;max-width:560px;margin:0 auto 8px;line-height:1.6;">
    A free, research-backed screening tool that estimates your child's risk of
    <strong style="color:#c5cae9;">Problematic Internet Use</strong> (PIU) in under 4 minutes.
  </div>
  <div style="font-size:0.82rem;color:#4a5568;margin-top:6px;">
    Powered by an ML ensemble trained on the Healthy Brain Network dataset &nbsp;·&nbsp; Not a clinical diagnosis
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;color:#c5cae9;margin-bottom:20px;'>What we measure</h3>", unsafe_allow_html=True)
    measures = [
        ("🧒","Demographics & Lifestyle","Age, sex, daily screen time, and how well your child copes with everyday life."),
        ("⚖️","Physical Health","Height, weight, BMI, resting heart rate, and blood pressure with automatic unit conversion."),
        ("🏃","Activity & Fitness","Days active per week, run endurance, push-ups, sit-ups, flexibility, grip strength, and trunk lift."),
        ("😴","Sleep Quality","Nightly sleep hours and quality mapped to a validated Sleep Disturbance Scale score."),
        ("⌚","Wearable / Actigraphy","Optional upload of fitness-tracker data for real movement stats — ENMO, wrist angle, and light exposure."),
        ("🧬","Body Composition","14 BIA-derived metrics auto-calculated from your physical measurements — no clinical device needed."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(measures):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:14px;padding:20px 18px;margin-bottom:16px;min-height:148px;">
  <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
  <div style="font-family:'Syne',sans-serif;font-weight:700;color:#e8eaf6;font-size:0.92rem;margin-bottom:6px;">{title}</div>
  <div style="color:#8892b0;font-size:0.83rem;line-height:1.55;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#c5cae9;margin-bottom:20px;'>The SII Scale</h3>", unsafe_allow_html=True)
    sii_items = [
        ("0","#00e5a0","None","No significant signs of problematic internet use."),
        ("1","#5c6fff","Mild","Some signs — monitor and consider lifestyle adjustments."),
        ("2","#ffb547","Moderate","Moderate concern — consider professional guidance."),
        ("3","#ff4d6d","Severe","Significant concern — professional support is strongly recommended."),
    ]
    sii_cols = st.columns(4)
    for i, (score, colour, label, desc) in enumerate(sii_items):
        with sii_cols[i]:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid {colour}33;
            border-radius:14px;padding:18px 14px;text-align:center;">
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{colour};line-height:1;">{score}</div>
  <div style="font-family:'Syne',sans-serif;font-weight:700;color:{colour};font-size:0.9rem;margin:6px 0 4px;">{label}</div>
  <div style="color:#8892b0;font-size:0.78rem;line-height:1.5;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    cta_col = st.columns([1, 2, 1])[1]
    with cta_col:
        if st.button("🚀  Start Assessment", key="home_start", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    st.markdown('<div style="text-align:center;color:#4a5568;font-size:0.75rem;margin-top:24px;">ScreenSight v2 &nbsp;·&nbsp; For informational purposes only &nbsp;·&nbsp; Not a substitute for professional medical advice</div>', unsafe_allow_html=True)


def render_step1():
    render_progress_bar(1)
    st.markdown("## About You / Your Child")
    answers = st.session_state.answers
    age = st.slider("How old are you / is your child?", min_value=5, max_value=22, step=1,
                    value=int(answers.get("Basic_Demos-Age", 10)))
    sex_options = ["Male", "Female"]
    sex_default = 0 if answers.get("Basic_Demos-Sex", 1) == 1 else 1
    sex_choice = st.radio("Are you / is your child male or female?", options=sex_options, index=sex_default, horizontal=True)
    sex_val = 1 if sex_choice == "Male" else 0
    coping = st.slider("On a scale of 1-10, how well is your child coping with day-to-day life?", min_value=1, max_value=10, step=1,
                       value=int(round((answers.get("CGAS-CGAS_Score", 60) - 25) / 7)) if "CGAS-CGAS_Score" in answers else 5)
    screen_time = st.slider("How many hours a day does your child spend on screens for fun?", min_value=0, max_value=16, step=1,
                            value=int(answers.get("PreInt_EduHx-computerinternet_hoursday", 3)))
    if st.button("Next ->", key="step1_next"):
        valid = True
        if not (5 <= age <= 22):
            st.error("Please enter an age between 5 and 22."); valid = False
        if not (0 <= screen_time <= 16):
            st.error("Screen time can't exceed 16 hours per day."); valid = False
        if valid:
            st.session_state.answers.update({
                "Basic_Demos-Age": float(age),
                "Basic_Demos-Sex": sex_val,
                "CGAS-CGAS_Score": float(coping * 7 + 25),
                "PreInt_EduHx-computerinternet_hoursday": float(screen_time),
            })
            st.session_state.step = 2; st.rerun()


def render_step2():
    render_progress_bar(2)
    st.markdown("## Physical Stats")
    answers = st.session_state.answers
    height_unit = st.radio("Height unit", ["cm", "ft / in"], horizontal=True, key="height_unit")
    if height_unit == "cm":
        height_cm = st.number_input("Height (cm)", min_value=90.0, max_value=220.0, step=0.5,
                                    value=float(answers.get("Physical-Height", 150.0)))
    else:
        col_ft, col_in = st.columns(2)
        stored_cm = answers.get("Physical-Height", 152.4)
        total_in = stored_cm / 2.54
        default_ft = max(2, min(7, int(total_in // 12)))
        default_in = max(0, min(11, int(total_in % 12)))
        with col_ft: feet = st.number_input("Feet", min_value=2, max_value=7, step=1, value=default_ft)
        with col_in: inches = st.number_input("Inches", min_value=0, max_value=11, step=1, value=default_in)
        height_cm = round((feet * 12 + inches) * 2.54, 1)
        st.caption(f"approx {height_cm} cm")
    weight_unit = st.radio("Weight unit", ["kg", "lbs"], horizontal=True, key="weight_unit")
    if weight_unit == "kg":
        weight_kg = st.number_input("Weight (kg)", min_value=15.0, max_value=150.0, step=0.5,
                                    value=float(answers.get("Physical-Weight", 50.0)))
    else:
        stored_kg = answers.get("Physical-Weight", 50.0)
        lbs = st.number_input("Weight (lbs)", min_value=33.0, max_value=330.0, step=0.5,
                              value=float(round(stored_kg / 0.453592, 1)))
        weight_kg = round(lbs * 0.453592, 1)
        st.caption(f"approx {weight_kg} kg")
    if height_cm > 0 and weight_kg > 0:
        bmi = round(weight_kg / (height_cm / 100) ** 2, 2)
        st.info(f"BMI: **{bmi}**")
    else:
        bmi = 0.0
    heart_rate = st.slider("Resting heart rate (bpm)", min_value=40, max_value=150, step=1,
                           value=int(answers.get("Physical-HeartRate", 75)))
    st.markdown("**Blood pressure** *(optional - leave at 0 to use age-appropriate defaults)*")
    col_sys, col_dia = st.columns(2)
    with col_sys: systolic = st.number_input("Systolic pressure (mmHg)", min_value=0, max_value=200, step=1,
                                              value=int(answers.get("Physical-Systolic_BP", 0)))
    with col_dia: diastolic = st.number_input("Diastolic pressure (mmHg)", min_value=0, max_value=130, step=1,
                                               value=int(answers.get("Physical-Diastolic_BP", 0)))
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("<- Back", key="step2_back"): st.session_state.step = 1; st.rerun()
    with col_next:
        if st.button("Next ->", key="step2_next"):
            valid = True
            if not (90 <= height_cm <= 220): st.error("Please enter a height between 90 cm and 220 cm."); valid = False
            if not (15 <= weight_kg <= 150): st.error("Please enter a weight between 15 kg and 150 kg."); valid = False
            if valid:
                age = int(answers.get("Basic_Demos-Age", 15))
                sex = int(answers.get("Basic_Demos-Sex", 1))
                activity_level = int(answers.get("BIA-BIA_Activity_Level_num", 3))
                if systolic <= 0: systolic = 110 if age < 13 else 118 if age < 18 else 120
                if diastolic <= 0: diastolic = 70 if age < 13 else 74 if age < 18 else 78
                bmi_val = round(weight_kg / (height_cm / 100) ** 2, 2)
                bia = derive_bia_fields(height_cm, weight_kg, age, sex, activity_level=activity_level)
                st.session_state.answers.update({
                    "Physical-Height": float(height_cm), "Physical-Weight": float(weight_kg),
                    "Physical-BMI": bmi_val, "BIA-BIA_BMI": bmi_val,
                    "Physical-HeartRate": float(heart_rate),
                    "Physical-Systolic_BP": float(systolic), "Physical-Diastolic_BP": float(diastolic),
                })
                st.session_state.answers.update(bia)
                st.session_state.step = 3; st.rerun()


def render_step3():
    render_progress_bar(3)
    st.markdown("## How Active Is Your Child?")
    answers = st.session_state.answers
    paq_stored = answers.get("PAQ_C-PAQ_C_Total", 2.0)
    paq_reverse = {v: k for k, v in _PAQ_MAP.items()}
    days_active = st.slider("How many days a week is your child physically active?", min_value=0, max_value=7, step=1,
                            value=paq_reverse.get(paq_stored, 3))
    paq_val = _PAQ_MAP[days_active]
    activity_options = ["Mostly sitting", "Light activity", "Moderate activity", "Active", "Very active"]
    act_default = max(0, int(answers.get("BIA-BIA_Activity_Level_num", 3)) - 1)
    activity_choice = st.radio("How would you describe your child's typical activity level?", options=activity_options, index=act_default)
    activity_level = activity_options.index(activity_choice) + 1
    endurance_options = ["Under 2 minutes", "2-5 minutes", "5-10 minutes", "10-15 minutes", "15+ minutes"]
    end_stage = int(answers.get("Fitness_Endurance-Max_Stage", 4))
    end_default = {2:0, 4:1, 7:2, 10:3, 14:4}.get(end_stage, 1)
    endurance_choice = st.radio("How long can your child run without stopping?", options=endurance_options, index=end_default)
    end_stage_val, end_mins_val, end_secs_val = _ENDURANCE_MAP[endurance_options.index(endurance_choice)]
    pushups = st.slider("How many push-ups can your child do?", min_value=0, max_value=50, step=1, value=int(answers.get("FGC-FGC_PU", 10)))
    situps = st.slider("How many sit-ups can your child do?", min_value=0, max_value=75, step=1, value=int(answers.get("FGC-FGC_CU", 20)))
    flex_options = ["Can't reach shins", "Reaches ankles", "Touches toes", "Past toes"]
    flex_stored = answers.get("FGC-FGC_SRL", 20)
    flex_default = _FLEX_MAP.index(flex_stored) if flex_stored in _FLEX_MAP else 1
    flex_choice = st.radio("How flexible is your child?", options=flex_options, index=flex_default)
    flex_val = _FLEX_MAP[flex_options.index(flex_choice)]
    grip_options = ["Weak", "Average", "Strong", "Very strong"]
    grip_stored = answers.get("FGC-FGC_GSD", 20)
    grip_default = _GRIP_MAP.index(grip_stored) if grip_stored in _GRIP_MAP else 1
    grip_choice = st.radio("How strong is your child's grip?", options=grip_options, index=grip_default)
    grip_val = _GRIP_MAP[grip_options.index(grip_choice)]
    trunk_lift = st.slider("How high can your child lift their trunk? (inches)", min_value=1, max_value=12, step=1, value=int(answers.get("FGC-FGC_TL", 6)))
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("<- Back", key="step3_back"): st.session_state.step = 2; st.rerun()
    with col_next:
        if st.button("Next ->", key="step3_next"):
            st.session_state.answers.update({
                "PAQ_C-PAQ_C_Total": paq_val, "PAQ_A-PAQ_A_Total": paq_val,
                "BIA-BIA_Activity_Level_num": activity_level,
                "Fitness_Endurance-Max_Stage": end_stage_val,
                "Fitness_Endurance-Time_Mins": end_mins_val, "Fitness_Endurance-Time_Sec": end_secs_val,
                "FGC-FGC_PU": pushups, "FGC-FGC_CU": situps,
                "FGC-FGC_SRL": float(flex_val), "FGC-FGC_SRR": float(flex_val),
                "FGC-FGC_GSD": float(grip_val), "FGC-FGC_GSND": float(grip_val - 2),
                "FGC-FGC_TL": trunk_lift,
            })
            h = st.session_state.answers.get("Physical-Height", 150.0)
            w = st.session_state.answers.get("Physical-Weight", 50.0)
            age = int(st.session_state.answers.get("Basic_Demos-Age", 15))
            sex = int(st.session_state.answers.get("Basic_Demos-Sex", 1))
            st.session_state.answers.update(derive_bia_fields(h, w, age, sex, activity_level=activity_level))
            st.session_state.step = 4; st.rerun()


def render_step4():
    render_progress_bar(4)
    st.markdown("## Sleep Habits")
    answers = st.session_state.answers
    raw_stored = answers.get("SDS-SDS_Total_Raw", 35)
    raw_to_hours = {v: k for k, v in _SLEEP_RAW_MAP.items()}
    hours_default = raw_to_hours.get(int(raw_stored), 8)
    sleep_hours = st.slider("How many hours does your child sleep on a typical night?", min_value=4, max_value=12, step=1, value=hours_default)
    raw_score = _SLEEP_RAW_MAP[sleep_hours]
    quality_options = ["Excellent", "Usually fine", "Often restless", "Poor"]
    quality_choice = st.radio("How would you describe your child's sleep quality?", options=quality_options, index=0)
    quality_offset = _SLEEP_QUALITY_OFFSETS[quality_options.index(quality_choice)]
    final_raw = min(72, raw_score + quality_offset)
    sds_t = round(final_raw * 0.8 + 14, 2)
    st.caption(f"Sleep disturbance score: {final_raw} (T-score: {sds_t})")
    screens_options = ["Yes", "No"]
    screens_default = 0 if answers.get("_bedtime_screens", False) else 1
    screens_choice = st.radio("Does your child use screens within 1 hour of bedtime?", options=screens_options, index=screens_default, horizontal=True)
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("<- Back", key="step4_back"): st.session_state.step = 3; st.rerun()
    with col_next:
        if st.button("Next ->", key="step4_next"):
            st.session_state.answers.update({
                "SDS-SDS_Total_Raw": float(final_raw), "SDS-SDS_Total_T": sds_t,
                "_bedtime_screens": screens_choice == "Yes",
            })
            st.session_state.step = 5; st.rerun()


def render_step5():
    render_progress_bar(5)
    st.markdown("## Wearable / Watch Data")
    st.markdown("Upload your child's fitness tracker data for a more accurate result.")
    uploaded_file = st.file_uploader("Upload your wearable data file", type=["csv", "xlsx", "parquet"],
                                     help="Accepts .csv, .xlsx, or .parquet files from Garmin, Fitbit, Apple Watch, etc.")
    if uploaded_file is not None:
        try:
            result = parse_watch_file(uploaded_file)
            values = result["values"]
            st.session_state.answers.update(values)
            st.session_state.watch_format = result["format"]
            st.session_state.watch_estimated = result["estimated"]
            st.markdown('<div style="color:#00e5a0;font-weight:700">Actigraphy auto-filled</div>', unsafe_allow_html=True)
            if result["estimated"]:
                st.warning("These values are estimated from summary data. For best accuracy, upload raw accelerometer data.")
            enmo_mean = values.get("enmo_mean", 0)
            enmo_std = values.get("enmo_std", 0)
            sparkline = np.clip(np.random.normal(enmo_mean, max(enmo_std, 0.001), 50), 0, None)
            st.line_chart({"Movement intensity": sparkline})
            metric_keys = ["enmo_mean", "anglez_mean", "X_mean", "Y_mean", "Z_mean", "light_mean"]
            cols = st.columns(3)
            for i, key in enumerate(metric_keys):
                with cols[i % 3]:
                    st.metric(label=ACTIGRAPHY_LABELS.get(key, key), value=f"{values.get(key, 0):.3f}")
            st.session_state.step = 6; st.rerun()
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("Could not read the file. Please check it is not corrupted and try again.")
    col_back, col_skip = st.columns(2)
    with col_back:
        if st.button("<- Back", key="step5_back"): st.session_state.step = 4; st.rerun()
    with col_skip:
        if st.button("Skip - I don't have a wearable", key="step5_skip"):
            st.session_state.answers.update(ACTIGRAPHY_DEFAULTS)
            st.session_state.watch_format = None
            st.session_state.watch_estimated = False
            st.session_state.step = 6; st.rerun()


def render_review_card(answers: dict):
    REVIEW_LABELS = {
        "Basic_Demos-Age": "Age", "Basic_Demos-Sex": "Sex",
        "CGAS-CGAS_Score": "Day-to-day coping (1-10 scale)",
        "PreInt_EduHx-computerinternet_hoursday": "Screen time (hours/day)",
        "Physical-Height": "Height (cm)", "Physical-Weight": "Weight (kg)",
        "Physical-BMI": "BMI", "Physical-HeartRate": "Resting heart rate (bpm)",
        "Physical-Systolic_BP": "Systolic BP (mmHg)", "Physical-Diastolic_BP": "Diastolic BP (mmHg)",
        "PAQ_C-PAQ_C_Total": "Days active per week (score)",
        "BIA-BIA_Activity_Level_num": "Activity level",
        "Fitness_Endurance-Max_Stage": "Run endurance stage",
        "FGC-FGC_PU": "Push-ups", "FGC-FGC_CU": "Sit-ups",
        "FGC-FGC_SRL": "Flexibility (cm)", "FGC-FGC_GSD": "Grip strength (kg)",
        "FGC-FGC_TL": "Trunk lift (inches)",
        "SDS-SDS_Total_Raw": "Sleep disturbance score", "_bedtime_screens": "Screens before bed?",
        "enmo_mean": "Movement intensity - avg", "anglez_mean": "Wrist angle - avg", "light_mean": "Ambient light - avg",
    }
    SECTIONS = [
        ("About", 1, ["Basic_Demos-Age","Basic_Demos-Sex","CGAS-CGAS_Score","PreInt_EduHx-computerinternet_hoursday"]),
        ("Physical", 2, ["Physical-Height","Physical-Weight","Physical-BMI","Physical-HeartRate","Physical-Systolic_BP","Physical-Diastolic_BP"]),
        ("Activity", 3, ["PAQ_C-PAQ_C_Total","BIA-BIA_Activity_Level_num","Fitness_Endurance-Max_Stage","FGC-FGC_PU","FGC-FGC_CU","FGC-FGC_SRL","FGC-FGC_GSD","FGC-FGC_TL"]),
        ("Sleep", 4, ["SDS-SDS_Total_Raw","_bedtime_screens"]),
        ("Watch", 5, ["enmo_mean","anglez_mean","light_mean"]),
    ]
    for section_name, step_num, keys in SECTIONS:
        col_title, col_edit = st.columns([4, 1])
        with col_title: st.markdown(f"### {section_name}")
        with col_edit:
            if st.button("Edit", key=f"edit_step{step_num}"):
                st.session_state.step = step_num; st.rerun()
        for key in keys:
            if key not in answers: continue
            label = REVIEW_LABELS.get(key, key)
            val = answers[key]
            if key == "Basic_Demos-Sex": display = "Male" if val == 1 else "Female"
            elif key == "_bedtime_screens": display = "Yes" if val else "No"
            elif key == "CGAS-CGAS_Score": display = f"{round((float(val)-25)/7)}/10"
            elif isinstance(val, float): display = f"{val:.2f}"
            else: display = str(val)
            st.markdown(f"**{label}:** {display}")
            st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(92,111,255,0.3),rgba(255,255,255,0.06),transparent);margin:20px 0 28px;border-radius:999px;"></div>', unsafe_allow_html=True)


def render_step6():
    render_progress_bar(6)
    st.markdown("## Review Your Answers")
    st.markdown("Check everything looks right before we run the assessment.")
    render_review_card(st.session_state.answers)
    col_back, col_run = st.columns(2)
    with col_back:
        if st.button("<- Back", key="step6_back"): st.session_state.step = 5; st.rerun()
    with col_run:
        if st.button("Run Assessment", key="step6_run", type="primary"):
            st.session_state.step = 7; st.rerun()
