"""
screensight/ui/theme.py
───────────────────────
inject_css() — injects the full dark glassmorphism theme into Streamlit.
Call once at app startup before any other st.* calls.
"""

import streamlit as st


def inject_css() -> None:
    """Inject global CSS: fonts, background, glass cards, badges, widgets, animations."""
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Syne:wght@600;700;800&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background: #080c1a !important;
        color: #e8eaf6;
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        color: #fff;
        letter-spacing: -0.02em;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(92,111,255,0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(92,111,255,0.7); }

    /* Animated gradient mesh */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        background:
            radial-gradient(ellipse 80% 60% at 20% 10%, rgba(92,111,255,0.22) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,229,160,0.10) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 60% 30%, rgba(255,77,109,0.07) 0%, transparent 50%);
        animation: meshMove 12s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes meshMove {
        0%   { background-position: 0% 0%, 100% 100%, 50% 30%; }
        33%  { background-position: 10% 20%, 85% 75%, 60% 20%; }
        66%  { background-position: 5% 15%, 90% 85%, 45% 40%; }
        100% { background-position: 15% 5%, 80% 90%, 55% 25%; }
    }

    /* Floating orbs */
    .orb { position:fixed; border-radius:50%; filter:blur(80px); pointer-events:none; z-index:0; opacity:0.18; }
    .orb-1 { width:160px; height:160px; background:#5c6fff; top:15%; left:8%;
              animation: orbDrift1 20s ease-in-out infinite alternate; }
    .orb-2 { width:120px; height:120px; background:#00e5a0; bottom:20%; right:10%;
              animation: orbDrift2 24s ease-in-out infinite alternate; }
    @keyframes orbDrift1 { 0% { transform:translate(0,0); } 100% { transform:translate(40px,60px); } }
    @keyframes orbDrift2 { 0% { transform:translate(0,0); } 100% { transform:translate(-50px,-40px); } }

    /* Glass card */
    .glass-card {
        background: rgba(255,255,255,0.04) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 24px 28px !important;
        margin-bottom: 20px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .glass-card:hover {
        border-color: rgba(92,111,255,0.4) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 0 0 1px rgba(92,111,255,0.15) !important;
    }

    /* SII badges */
    .badge-none     { display:inline-block; background:linear-gradient(135deg,#00e5a0,#00c48c); color:#080c1a;
                      font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem;
                      padding:4px 14px; border-radius:999px; box-shadow:0 0 12px rgba(0,229,160,0.45); }
    .badge-mild     { display:inline-block; background:linear-gradient(135deg,#5c6fff,#4455ee); color:#fff;
                      font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem;
                      padding:4px 14px; border-radius:999px; box-shadow:0 0 12px rgba(92,111,255,0.45); }
    .badge-moderate { display:inline-block; background:linear-gradient(135deg,#ffb547,#f59e0b); color:#080c1a;
                      font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem;
                      padding:4px 14px; border-radius:999px; box-shadow:0 0 12px rgba(255,181,71,0.45); }
    .badge-severe   { display:inline-block; background:linear-gradient(135deg,#ff4d6d,#e0284a); color:#fff;
                      font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem;
                      padding:4px 14px; border-radius:999px; box-shadow:0 0 12px rgba(255,77,109,0.45); }

    /* Progress strip */
    .progress-strip { display:flex; align-items:center; justify-content:center; gap:0; margin:24px 0 8px; }
    .progress-node  { width:36px; height:36px; border-radius:50%; background:rgba(255,255,255,0.06);
                      border:2px solid rgba(255,255,255,0.15); display:flex; align-items:center;
                      justify-content:center; font-family:'Syne',sans-serif; font-weight:700;
                      font-size:0.85rem; color:rgba(255,255,255,0.4); position:relative; z-index:2;
                      flex-shrink:0;
                      transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s ease; }
    .progress-node.active { background:#5c6fff; border-color:#5c6fff; color:#fff; transform:scale(1.2);
                            box-shadow:0 0 0 4px rgba(92,111,255,0.2), 0 0 0 8px rgba(92,111,255,0.08); }
    .progress-node.done   { background:#00e5a0; border-color:#00e5a0; color:#080c1a; }
    .progress-line { flex:1; height:2px; background:rgba(255,255,255,0.08); position:relative;
                     overflow:hidden; min-width:20px; max-width:60px; }
    .progress-line::after { content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
                            background:linear-gradient(90deg,transparent,#5c6fff,transparent);
                            animation:lineFlow 2s linear infinite; }
    @keyframes lineFlow { 0% { left:-100%; } 100% { left:100%; } }
    .step-chip { display:inline-block; font-family:'DM Sans',sans-serif; font-size:0.72rem;
                 font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:#8892b0;
                 background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);
                 border-radius:999px; padding:3px 12px; margin-top:8px; }

    /* Fade-up entrance */
    @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
    [data-testid="block-container"] { animation:fadeUp 0.4s ease both; padding-top:2rem; position:relative; z-index:1; }

    /* Ripple */
    .next-btn { position:relative; overflow:hidden; }
    .next-btn:active::after { content:''; position:absolute; top:50%; left:50%; width:0; height:0;
                              background:rgba(255,255,255,0.3); border-radius:50%;
                              transform:translate(-50%,-50%); animation:ripple 0.5s ease-out forwards; }
    @keyframes ripple { to { width:200px; height:200px; opacity:0; } }

    /* Widget overrides */
    [data-testid="stSlider"] [data-testid="stSliderThumb"] { background:#5c6fff !important; border:2px solid #fff !important; box-shadow:0 0 8px rgba(92,111,255,0.6) !important; }
    [data-testid="stSlider"] [role="slider"] { background:#5c6fff !important; }
    [data-testid="stRadio"] label { transition:color 0.15s ease; }
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { font-family:'DM Sans',sans-serif !important; }
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background:rgba(255,255,255,0.05) !important; border:1px solid rgba(255,255,255,0.12) !important;
        border-radius:8px !important; color:#e8eaf6 !important; font-family:'DM Sans',sans-serif !important;
        transition:border-color 0.2s ease, box-shadow 0.2s ease !important; }
    [data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus {
        border-color:rgba(92,111,255,0.6) !important; box-shadow:0 0 0 2px rgba(92,111,255,0.25) !important; outline:none !important; }
    [data-testid="stButton"] button[kind="primary"] {
        background:linear-gradient(135deg,#5c6fff,#4455ee) !important; border:none !important;
        border-radius:10px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important;
        transition:transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow:0 4px 16px rgba(92,111,255,0.35) !important; }
    [data-testid="stButton"] button[kind="primary"]:hover { transform:translateY(-1px) !important; box-shadow:0 6px 24px rgba(92,111,255,0.5) !important; }
    [data-testid="stButton"] button[kind="secondary"], [data-testid="stButton"] button:not([kind]) {
        background:rgba(255,255,255,0.06) !important; border:1px solid rgba(255,255,255,0.12) !important;
        border-radius:10px !important; color:#c5cae9 !important; font-family:'DM Sans',sans-serif !important;
        transition:background 0.2s ease, border-color 0.2s ease !important; }
    [data-testid="stButton"] button[kind="secondary"]:hover, [data-testid="stButton"] button:not([kind]):hover {
        background:rgba(255,255,255,0.10) !important; border-color:rgba(92,111,255,0.35) !important; }
    [data-testid="stDownloadButton"] button {
        background:linear-gradient(135deg,#00e5a0,#00c48c) !important; border:none !important;
        border-radius:10px !important; color:#080c1a !important; font-family:'Syne',sans-serif !important;
        font-weight:700 !important; transition:transform 0.15s ease, box-shadow 0.15s ease !important;
        box-shadow:0 4px 16px rgba(0,229,160,0.3) !important; }
    [data-testid="stDownloadButton"] button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 24px rgba(0,229,160,0.5) !important; }
    [data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid rgba(255,255,255,0.08) !important; gap:4px !important; }
    [data-testid="stTabs"] [role="tab"] { background:transparent !important; border:none !important;
        border-bottom:2px solid transparent !important; color:#8892b0 !important;
        font-family:'DM Sans',sans-serif !important; font-weight:500 !important; padding:8px 16px !important;
        transition:color 0.2s ease, border-color 0.2s ease !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#5c6fff !important; border-bottom-color:#5c6fff !important; }
    [data-testid="stTabs"] [role="tab"]:hover { color:#c5cae9 !important; }
    [data-testid="stFileUploader"] { border:2px dashed rgba(92,111,255,0.35) !important; border-radius:12px !important;
        background:rgba(92,111,255,0.04) !important; transition:border-color 0.2s ease, background 0.2s ease !important; }
    [data-testid="stFileUploader"]:hover { border-color:rgba(92,111,255,0.65) !important; background:rgba(92,111,255,0.08) !important; }
    [data-testid="stMetric"] { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:12px !important; padding:12px 16px !important; }
    [data-testid="stMetricValue"] { color:#5c6fff !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; }
    [data-testid="stExpander"] { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:12px !important; }
    [data-testid="stAlert"] { border-radius:10px !important; backdrop-filter:blur(8px) !important; }
    [data-testid="stAlert"][data-baseweb="notification"] { background:rgba(92,111,255,0.08) !important; border-left:3px solid #5c6fff !important; }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="orb orb-1"></div><div class="orb orb-2"></div>',
        unsafe_allow_html=True,
    )
