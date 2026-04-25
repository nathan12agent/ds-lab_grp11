"""
app.py - ScreenSight v2 entry point
------------------------------------
Thin router: page config, CSS, session state, then dispatch to step renderers.

Module layout
-------------
screensight/
  constants.py          FEATURE_COLS, SII_LABELS, ACTIGRAPHY_*, MODEL_ACCURACY
  wearable.py           derive_bia_fields, parse_watch_file, export_report
  ml.py                 train_models, build_input_row, run_inference
  ui/
    theme.py            inject_css()
    navigation.py       init_session_state(), render_progress_bar()
    steps.py            render_home(), render_step1() ... render_step6()
    results.py          render_step7()
"""
import os
import warnings
warnings.filterwarnings("ignore")

_TESTING = os.environ.get("PYTEST_CURRENT_TEST") is not None

if not _TESTING:
    import streamlit as st
    st.set_page_config(page_title="ScreenSight v2", page_icon="🧠", layout="wide")

    from screensight.ui.theme      import inject_css
    from screensight.ui.navigation import init_session_state
    from screensight.ui.steps      import (
        render_home, render_step1, render_step2,
        render_step3, render_step4, render_step5, render_step6,
    )
    from screensight.ui.results    import render_step7

    inject_css()
    init_session_state()

    step = st.session_state.step
    _, main_col, _ = st.columns([1, 6, 1])

    with main_col:
        if   step == 0: render_home()
        elif step == 1: render_step1()
        elif step == 2: render_step2()
        elif step == 3: render_step3()
        elif step == 4: render_step4()
        elif step == 5: render_step5()
        elif step == 6: render_step6()
        elif step == 7: render_step7()

# ── Test-mode imports (no Streamlit) ─────────────────────────────────────────
# The test suite imports these directly from the screensight package.
# Keep these re-exports so existing tests (which import from app) still work.
from screensight.constants import (          # noqa: E402
    FEATURE_COLS, ACTIGRAPHY_DEFAULTS, ACTIGRAPHY_LABELS, SII_LABELS,
)
from screensight.wearable import (           # noqa: E402
    derive_bia_fields, parse_watch_file, export_report,
)
