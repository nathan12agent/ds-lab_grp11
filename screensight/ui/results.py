"""
screensight/ui/results.py
-------------------------
render_step7() - loads ensemble pkl, shows 5-model vote cards,
4 chart tabs, SII explainer, download button.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from screensight.constants import SII_LABELS, MODEL_ACCURACY, MODEL_WEIGHTS
from screensight.ensemble_predict import run_inference, _ensemble
from screensight.wearable import export_report
from screensight.ui.navigation import render_progress_bar


def render_step7():
    render_progress_bar(7)
    st.markdown("## Your Assessment Results")
    answers = st.session_state.answers

    try:
       predictions, final_pred, input_df, confidence, conflict = run_inference(answers)
    except Exception as e:
        st.error(f"Could not load the ensemble model. Run: python screensight/train_ensemble.py\n\nDetail: {e}")
        st.stop()
        return

    st.session_state.result = final_pred

    sii_label, _, sii_desc = SII_LABELS[final_pred]
    badge_colours = {0: "#00e5a0", 1: "#5c6fff", 2: "#ffb547", 3: "#ff4d6d"}
    badge_colour  = badge_colours[final_pred]

    # ── Stat strip ────────────────────────────────────────────────────────────
    age_val = int(answers.get("Basic_Demos-Age", 0))
    sex_val = "M" if answers.get("Basic_Demos-Sex", 1) == 1 else "F"
    scr_val = answers.get("PreInt_EduHx-computerinternet_hoursday", 0)
    bmi_val = answers.get("Physical-BMI", 0)
    sds_val = answers.get("SDS-SDS_Total_T", 0)
    paq_val = answers.get("PAQ_C-PAQ_C_Total", 0)
    stat_items = [
        ("Age / Sex",         f"{age_val} - {sex_val}",      "#c5cae9"),
        ("Screen time",       f"{scr_val:.0f} hrs/day",      "#ffb547" if scr_val > 4 else "#00e5a0"),
        ("BMI",               f"{bmi_val:.1f}",              "#c5cae9"),
        ("Sleep disturbance", f"T-score {sds_val:.0f}",      "#ff4d6d" if sds_val > 65 else "#ffb547" if sds_val > 55 else "#00e5a0"),
        ("Activity score",    f"{paq_val:.1f} / 5",          "#00e5a0" if paq_val >= 3 else "#ffb547"),
    ]
    stat_cols = st.columns(5)
    for col, (label, value, colour) in zip(stat_cols, stat_items):
        with col:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:14px 12px;text-align:center;margin-bottom:20px;">
  <div style="font-size:0.7rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">{label}</div>
  <div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{colour};">{value}</div>
</div>""", unsafe_allow_html=True)

    # ── SII result card ───────────────────────────────────────────────────────
    count_up_js = f"""<script>(function(){{var el=document.getElementById('sii-number');if(!el)return;var t={final_pred},c=0;var iv=setInterval(function(){{el.textContent=c;if(c>=t)clearInterval(iv);c++;}},120);}})();</script>"""
    st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:16px;padding:28px 32px;margin-bottom:24px;backdrop-filter:blur(12px);">
  <div style="font-family:Syne,sans-serif;font-size:0.85rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Severity Index (SII)</div>
  <div id="sii-number" style="font-family:Syne,sans-serif;font-size:5rem;font-weight:800;color:{badge_colour};line-height:1;">{final_pred}</div>
  <span style="display:inline-block;background:{badge_colour};color:#080c1a;font-family:Syne,sans-serif;font-weight:700;font-size:1rem;padding:5px 18px;border-radius:999px;margin:10px 0 12px;">{sii_label}</span>
  <p style="color:#b0bec5;font-size:0.95rem;margin-top:4px;">{sii_desc}</p>
</div>{count_up_js}""", unsafe_allow_html=True)

# ── Confidence banner ─────────────────────────────────────────────────────
    if conflict:
        st.warning(
            f"⚠️ Models show conflicting signals — prediction confidence is lower. "
            f"Meta-learner confidence: {confidence*100:.0f}%. "
            f"Consider consulting a professional."
        )
    else:
        st.success(f"✓ Strong model agreement — confidence: {confidence*100:.0f}%")

    
    # ── 5-model vote cards ────────────────────────────────────────────────────
    st.markdown("### Ensemble Model Predictions")
    st.caption("Stacking ensemble of 5 classifiers: ET · SVM · RF · GB · XGB. Final prediction determined by tuned XGBoost meta-learner.")

    model_cols = st.columns(5)
    for i, (name, pred) in enumerate(predictions.items()):
        pred_label  = SII_LABELS[pred][0]
        pred_colour = badge_colours[pred]
        acc_badge   = MODEL_ACCURACY.get(name, "-")
        border      = "rgba(92,111,255,0.5)" if pred == final_pred else "rgba(255,255,255,0.08)"
        shadow      = "0 0 0 1px rgba(92,111,255,0.2)" if pred == final_pred else "none"
        with model_cols[i]:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid {border};border-radius:12px;
            padding:16px 10px;text-align:center;box-shadow:{shadow};">
  <div style="font-size:0.68rem;color:#8892b0;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
  <div style="font-size:0.62rem;color:#4a5568;margin-bottom:8px;">acc {acc_badge}</div>
  <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{pred_colour};line-height:1;">{pred}</div>
  <div style="display:inline-block;background:{pred_colour};color:#080c1a;font-family:Syne,sans-serif;font-weight:700;font-size:0.7rem;padding:2px 10px;border-radius:999px;margin-top:6px;">{pred_label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk scores (shared across tabs 1 and 2) ──────────────────────────────
    risk_keys   = ["PreInt_EduHx-computerinternet_hoursday","SDS-SDS_Total_Raw","PAQ_C-PAQ_C_Total","Fitness_Endurance-Max_Stage","BIA-BIA_BMI","CGAS-CGAS_Score"]
    risk_labels = ["Screen time","Sleep quality","Physical activity","Fitness level","Body composition","Coping ability"]
    risk_ranges = {
        "PreInt_EduHx-computerinternet_hoursday": (0, 16),
        "SDS-SDS_Total_Raw":                      (24, 72),
        "PAQ_C-PAQ_C_Total":                      (5.0, 1.0),
        "Fitness_Endurance-Max_Stage":            (14, 2),
        "BIA-BIA_BMI":                            (18.5, 35),
        "CGAS-CGAS_Score":                        (95, 32),
    }
    risk_scores = []
    for key in risk_keys:
        val = answers.get(key, 0)
        lo, hi = risk_ranges[key]
        score = max(0.0, min(1.0, (float(val)-lo)/(hi-lo))) if hi != lo else 0.0
        risk_scores.append(round(score*100, 1))

    tab1, tab2, tab3, tab4 = st.tabs(["Risk Breakdown", "Radar Profile", "Model Confidence", "Improvement Areas"])

    with tab1:
        st.caption("How each area contributes to the overall risk score.")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        fig1.patch.set_facecolor("#0d1226"); ax1.set_facecolor("#0d1226")
        colours = ["#ff4d6d" if s>66 else "#ffb547" if s>33 else "#00e5a0" for s in risk_scores]
        bars = ax1.barh(risk_labels, risk_scores, color=colours, height=0.5, edgecolor="none")
        ax1.set_xlim(0, 115)
        ax1.set_xlabel("Risk level (%)", color="#8892b0", fontsize=9)
        ax1.tick_params(axis="both", colors="#c5cae9", labelsize=9)
        for spine in ax1.spines.values(): spine.set_visible(False)
        ax1.axvline(33, color=(1,1,1,0.06), linewidth=1, linestyle="--")
        ax1.axvline(66, color=(1,1,1,0.06), linewidth=1, linestyle="--")
        for bar, score, colour in zip(bars, risk_scores, colours):
            ax1.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2, f"{score:.0f}%", va="center", color=colour, fontsize=9, fontweight="bold")
        fig1.tight_layout(pad=1.5); st.pyplot(fig1); plt.close(fig1)

    with tab2:
        st.caption("A holistic view of six key dimensions.")
        radar_labels = ["Screen time","Sleep","Activity","Fitness","Body health","Coping"]
        rv = [s/100 for s in risk_scores] + [risk_scores[0]/100]
        angles = [n/float(len(radar_labels))*2*np.pi for n in range(len(radar_labels))] + [0]
        fig2, ax2 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#0d1226"); ax2.set_facecolor("#0d1226")
        ax2.plot(angles, rv, color="#5c6fff", linewidth=2.5)
        ax2.fill(angles, rv, color="#5c6fff", alpha=0.20)
        ax2.scatter(angles[:-1], rv[:-1], color="#5c6fff", s=50, zorder=5)
        ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(radar_labels, color="#c5cae9", fontsize=9)
        ax2.set_yticklabels([]); ax2.set_ylim(0,1)
        ax2.spines["polar"].set_color((1.0,1.0,1.0,0.08)); ax2.grid(color=(1.0,1.0,1.0,0.08), linewidth=0.8)
        fig2.tight_layout(pad=1.5); st.pyplot(fig2); plt.close(fig2)

    with tab3:
        st.caption("Per-model class probabilities across all 5 classifiers.")
        model_names  = list(_ensemble.named_estimators_.keys())
        class_labels = ["None", "Mild", "Moderate", "Severe"]
        sii_classes  = [0, 1, 2, 3]
        prob_matrix  = np.zeros((len(model_names), 4))
        for i, (name, estimator) in enumerate(_ensemble.named_estimators_.items()):
            try:
                probs = estimator.predict_proba(input_df)[0]
                for j, cls in enumerate(estimator.classes_):
                    if cls in sii_classes:
                        prob_matrix[i, sii_classes.index(int(cls))] = probs[j]
            except Exception:
                prob_matrix[i, predictions.get(name, final_pred)] = 1.0

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fig3.patch.set_facecolor("#0d1226"); ax3.set_facecolor("#0d1226")
        x = np.arange(len(model_names)); width = 0.18
        bar_colours = ["#00e5a0", "#5c6fff", "#ffb547", "#ff4d6d"]
        for j, (cls_label, colour) in enumerate(zip(class_labels, bar_colours)):
            ax3.bar(x+(j-1.5)*width, prob_matrix[:,j], width, label=cls_label, color=colour, alpha=0.88, edgecolor="none")
        ax3.set_xticks(x); ax3.set_xticklabels(model_names, color="#c5cae9", fontsize=8.5)
        ax3.set_ylabel("Probability", color="#8892b0", fontsize=9); ax3.set_ylim(0, 1.05)
        ax3.tick_params(axis="both", colors="#c5cae9", labelsize=8)
        for spine in ax3.spines.values(): spine.set_visible(False)
        ax3.yaxis.grid(True, color=(1.0,1.0,1.0,0.06), linewidth=0.8); ax3.set_axisbelow(True)
        ax3.legend(fontsize=8.5, facecolor="#0d1226", edgecolor=(1,1,1,0.1), labelcolor="#c5cae9", framealpha=0.9)
        fig3.tight_layout(pad=1.5); st.pyplot(fig3); plt.close(fig3)

    with tab4:
        st.caption("Personalised suggestions based on your answers.")
        recs = []
        screen_h = float(answers.get("PreInt_EduHx-computerinternet_hoursday", 0))
        if screen_h > 4: recs.append(("Screen time", f"Your child spends {screen_h:.0f} hrs/day on screens. Try a 2-3 hour daily limit."))
        elif screen_h > 2: recs.append(("Screen time", "Screen time is moderate. Consider screen-free evenings."))
        sds_raw = float(answers.get("SDS-SDS_Total_Raw", 35))
        if sds_raw >= 55: recs.append(("Sleep", "Sleep quality appears poor. A consistent bedtime routine and limiting screens before bed can help."))
        elif sds_raw >= 45: recs.append(("Sleep", "Sleep could be improved. Aim for 8-10 hours and avoid caffeine after 3 pm."))
        paq = float(answers.get("PAQ_C-PAQ_C_Total", 2.5))
        if paq < 2.0: recs.append(("Physical activity", "Your child is not very active. Even 30 minutes of walking per day reduces PIU risk."))
        elif paq < 3.0: recs.append(("Physical activity", "Moderate activity. Encourage team sports or outdoor activities."))
        cgas = float(answers.get("CGAS-CGAS_Score", 60))
        if cgas < 50: recs.append(("Coping skills", "Coping seems challenging. Consider speaking with a school counsellor or family therapist."))
        if answers.get("_bedtime_screens", False): recs.append(("Bedtime screens", "Screens before bed disrupt melatonin. Try a no-screens-after-9pm rule."))
        if not recs: recs.append(("Looking good!", "No major risk areas identified. Check in again in 3-6 months."))
        for title, text in recs:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 20px;margin-bottom:12px;">
  <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#e8eaf6;margin-bottom:6px;">{title}</div>
  <div style="color:#b0bec5;font-size:0.9rem;line-height:1.55;">{text}</div></div>""", unsafe_allow_html=True)

    with st.expander("What does SII mean?"):
        st.markdown("""**SII (Severity Index)** is a 0-3 scale:
| Score | Label | Meaning |
|---|---|---|
| 0 | None | No significant signs |
| 1 | Mild | Some signs - monitor |
| 2 | Moderate | Consider professional guidance |
| 3 | Severe | Professional support strongly recommended |

Produced by a stacking ensemble of 5 classifiers (ET, SVM, RF, GB, XGB) with a tuned XGBoost meta-learner **Not a clinical diagnosis.**""")

    report_html = export_report(answers, final_pred, predictions)
    st.download_button(label="Download Report", data=report_html.encode("utf-8"),
                       file_name="screensight_report.html", mime="text/html")

    if st.button("Start Over", key="start_over"):
        st.session_state.step = 0; st.session_state.answers = {}
        st.session_state.result = None; st.session_state.watch_format = None
        st.session_state.watch_estimated = False; st.rerun()
