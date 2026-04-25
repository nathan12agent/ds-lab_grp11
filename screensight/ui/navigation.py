"""
screensight/ui/navigation.py
────────────────────────────
Session state initialisation and the stepped progress bar.
"""

import streamlit as st

STEP_LABELS = ["About", "Physical", "Activity", "Sleep", "Watch", "Review", "Results"]


def init_session_state() -> None:
    """Initialise all session state keys with defaults (idempotent)."""
    defaults = {
        "step":            0,
        "answers":         {},
        "result":          None,
        "watch_format":    None,
        "watch_estimated": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_progress_bar(current_step: int) -> None:
    """
    Render the 7-node stepped progress strip with an animated connecting line
    and a 'STEP N OF 7 · Label' chip below it.

    Args:
        current_step: 1–7 (steps 1–7 of the assessment flow)
    """
    nodes_html = ""
    for i, label in enumerate(STEP_LABELS, start=1):
        if i < current_step:
            cls, icon = "done", "✓"
        elif i == current_step:
            cls, icon = "active", str(i)
        else:
            cls, icon = "", str(i)

        nodes_html += f'<div class="progress-node {cls}" title="{label}">{icon}</div>'
        if i < 7:
            nodes_html += '<div class="progress-line"></div>'

    chip = (
        f'<div style="text-align:center;margin-top:10px;margin-bottom:24px;">'
        f'<span class="step-chip">STEP {current_step} OF 7 &nbsp;·&nbsp; '
        f'{STEP_LABELS[current_step - 1]}</span></div>'
    )

    st.markdown(f'<div class="progress-strip">{nodes_html}</div>{chip}',
                unsafe_allow_html=True)
