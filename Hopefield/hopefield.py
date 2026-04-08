"""
🧠 Hopfield Network — Interactive Learning Module
A beginner-friendly, visually stunning Streamlit app.
Run with: streamlit run hopfield_learning_module.py
Requirements: streamlit, numpy, pandas, plotly
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="Hopfield Network · Neural Memory Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except st.errors.StreamlitAPIException:
    pass
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Bioluminescent Deep-Sea Theme
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne+Mono&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-void:     #020817;
    --bg-deep:     #050f1f;
    --bg-card:     #0a1628;
    --bg-card2:    #0d1f38;
    --border:      rgba(0, 212, 255, 0.15);
    --border-glow: rgba(0, 212, 255, 0.4);
    --cyan:        #00d4ff;
    --cyan-dim:    #0099bb;
    --teal:        #00ffcc;
    --electric:    #4f8cff;
    --amber:       #ffb347;
    --rose:        #ff5580;
    --text-bright: #e2f4ff;
    --text-mid:    #7ab8d4;
    --text-dim:    #3a6680;
    --mono:        'Syne Mono', monospace;
    --display:     'Syne', sans-serif;
    --body:        'Space Grotesk', sans-serif;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-void) !important;
    font-family: var(--body);
    color: var(--text-bright);
}
[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 4px; }

/* ── Hero Header ── */
.hero {
    position: relative;
    padding: 3rem 2.5rem 2.5rem;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0,255,204,0.05) 0%, transparent 50%),
                var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--teal), transparent);
    animation: scanline 3s ease-in-out infinite;
}
@keyframes scanline { 0%,100%{opacity:0.4} 50%{opacity:1} }

.hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    opacity: 0.8;
}
.hero-title {
    font-family: var(--display);
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2f4ff 30%, var(--cyan) 70%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-mid);
    max-width: 600px;
    line-height: 1.7;
}

/* ── Section Headers ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: var(--display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-bright);
    margin-bottom: 0.2rem;
}
.section-desc {
    font-size: 0.9rem;
    color: var(--text-mid);
    margin-bottom: 1.2rem;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.card:hover { border-color: var(--border-glow); }
.card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
}

.card-glow {
    box-shadow: 0 0 30px rgba(0,212,255,0.08), inset 0 1px 0 rgba(0,212,255,0.1);
}

/* ── Callout Boxes ── */
.callout {
    padding: 1.2rem 1.4rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 3px solid;
    font-size: 0.93rem;
    line-height: 1.65;
}
.callout-info {
    background: rgba(0,212,255,0.06);
    border-color: var(--cyan);
    color: var(--text-mid);
}
.callout-tip {
    background: rgba(0,255,204,0.06);
    border-color: var(--teal);
    color: var(--text-mid);
}
.callout-warn {
    background: rgba(255,179,71,0.06);
    border-color: var(--amber);
    color: #c8a070;
}
.callout-title {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.9;
}

/* ── Metric Pills ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    flex: 1; min-width: 120px;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-pill .val {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--cyan);
}
.metric-pill .lbl {
    font-size: 0.75rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Progress Steps ── */
.steps { display: flex; gap: 0; margin: 1.5rem 0; }
.step {
    flex: 1; text-align: center;
    padding: 0.7rem 0.4rem;
    border-top: 2px solid var(--bg-card2);
    font-size: 0.8rem;
    color: var(--text-dim);
    transition: 0.3s;
    cursor: default;
}
.step.active {
    border-color: var(--cyan);
    color: var(--cyan);
    font-weight: 600;
}
.step.done { border-color: var(--teal); color: var(--teal); }
.step-num {
    display: block;
    font-family: var(--mono);
    font-size: 0.65rem;
    opacity: 0.6;
    margin-bottom: 0.2rem;
}

/* ── Grid Neurons ── */
div.stButton > button {
    height: 42px !important;
    width: 42px !important;
    padding: 0 !important;
    min-height: 0 !important;
    border-radius: 6px !important;
    background: #0a1628 !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    color: transparent !important;
    transition: all 0.15s ease !important;
    font-size: 0 !important;
}
div.stButton > button:hover {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.25) !important;
    background: rgba(0,212,255,0.08) !important;
}
.neuron-on div.stButton > button {
    background: radial-gradient(circle, #00d4ff 0%, #0099cc 100%) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.6), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.neuron-on div.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,212,255,0.8) !important;
}

/* ── Sidebar Nav ── */
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--body);
    font-size: 0.9rem;
    color: var(--text-mid);
    padding: 0.3rem 0;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--cyan); }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--body) !important;
    font-size: 0.88rem !important;
    color: var(--text-dim) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--cyan); }

/* ── Markdown ── */
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {
    font-family: var(--display);
    color: var(--text-bright);
}
.stMarkdown code {
    font-family: var(--mono);
    background: rgba(0,212,255,0.08);
    color: var(--cyan);
    border-radius: 4px;
    padding: 0.1em 0.35em;
    font-size: 0.88em;
    border: 1px solid var(--border);
}
.stMarkdown blockquote {
    border-left: 3px solid var(--cyan);
    background: rgba(0,212,255,0.04);
    color: var(--text-mid);
    padding: 0.5rem 1rem;
    border-radius: 0 6px 6px 0;
}

/* ── Input elements ── */
[data-testid="stTextInput"] input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-bright) !important;
    font-family: var(--body) !important;
}
[data-testid="stSelectbox"] > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Primary button override ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0077aa, #00a8cc) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(0,168,204,0.3) !important;
    border-radius: 8px !important;
    height: 38px !important;
    width: auto !important;
    padding: 0 1.2rem !important;
    font-size: 0.88rem !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 28px rgba(0,212,255,0.5) !important;
    background: linear-gradient(135deg, #0099cc, #00ccee) !important;
}
div.stButton > button[kind="secondary"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
    border-radius: 8px !important;
    height: 38px !important;
    width: auto !important;
    padding: 0 1.2rem !important;
    font-size: 0.88rem !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2em 0.7em;
    border-radius: 99px;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-cyan { background: rgba(0,212,255,0.12); color: var(--cyan); border: 1px solid rgba(0,212,255,0.25); }
.badge-teal { background: rgba(0,255,204,0.1); color: var(--teal); border: 1px solid rgba(0,255,204,0.2); }
.badge-amber { background: rgba(255,179,71,0.1); color: var(--amber); border: 1px solid rgba(255,179,71,0.2); }

/* ── Pulse animation ── */
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 8px rgba(0,212,255,0.3); }
    50%      { box-shadow: 0 0 24px rgba(0,212,255,0.7); }
}
.pulse { animation: pulse-glow 2s ease-in-out infinite; }

/* ── Energy bar ── */
.energy-bar-wrap {
    background: var(--bg-card2);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
    border: 1px solid var(--border);
}
.energy-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
    background: linear-gradient(90deg, #00a8cc, #00d4ff, #00ffcc);
}

/* ── Log terminal ── */
.log-terminal {
    background: #010a14;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #4fa8c8;
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.8;
}
.log-line { margin: 0; opacity: 0.85; }
.log-line:last-child { color: var(--cyan); opacity: 1; }
.log-line::before { content: '› '; color: var(--teal); }

/* ── Concept cards ── */
.concept-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 1rem 0; }
.concept-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: 0.2s;
}
.concept-card:hover { border-color: var(--border-glow); transform: translateY(-2px); }
.concept-icon { font-size: 2rem; margin-bottom: 0.6rem; display: block; }
.concept-name {
    font-family: var(--display);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-bright);
    margin-bottom: 0.4rem;
}
.concept-desc { font-size: 0.78rem; color: var(--text-dim); line-height: 1.5; }

/* ── Plotly theme override ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Sidebar logo ── */
.sidebar-brand {
    font-family: var(--display);
    font-weight: 800;
    font-size: 1.1rem;
    color: var(--cyan);
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}
.sidebar-brand span { color: var(--teal); }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HOPFIELD NETWORK ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class HopfieldNetwork:
    def __init__(self, size=49):
        self.N = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        self.W = np.zeros((self.N, self.N))
        if not patterns:
            return
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        return -0.5 * float(state @ self.W @ state)

    def update_async(self, state, idx=None):
        s = state.copy()
        if idx is None:
            idx = np.random.randint(0, self.N)
        act = float(self.W[idx] @ s)
        s[idx] = 1 if act >= 0 else -1
        return s, idx, act

    def update_sync(self, state):
        return np.where(self.W @ state >= 0, 1, -1)

    def overlap(self, state, memory):
        """Cosine similarity with stored memory — 1 = perfect recall."""
        return float(state @ memory) / self.N

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
def b(grid):
    return np.array(grid).flatten().astype(float) * 2 - 1

DEFAULT_PATTERNS = {
    "Letter N": b([[1,0,0,0,0,0,1],[1,1,0,0,0,0,1],[1,0,1,0,0,0,1],[1,0,0,1,0,0,1],[1,0,0,0,1,0,1],[1,0,0,0,0,1,1],[1,0,0,0,0,0,1]]),
    "Letter Z": b([[1,1,1,1,1,1,1],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],[0,1,0,0,0,0,0],[1,1,1,1,1,1,1]]),
    "Diamond":  b([[0,0,0,1,0,0,0],[0,0,1,0,1,0,0],[0,1,0,0,0,1,0],[1,0,0,0,0,0,1],[0,1,0,0,0,1,0],[0,0,1,0,1,0,0],[0,0,0,1,0,0,0]]),
    "Square":   b([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,1,1,1,1,1,1]]),
    "Cross":    b([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[1,1,1,1,1,1,1],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]]),
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#7ab8d4"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def style_fig(fig, h=250, title=""):
    fig.update_layout(**PLOT_LAYOUT, height=h, title=dict(text=title, font=dict(color="#e2f4ff", size=13)))
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)", tickfont=dict(size=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def hero(title, subtitle, eyebrow="", badge=""):
    badge_html = f'<span class="badge badge-cyan" style="margin-bottom:1rem;display:inline-block;">{badge}</span><br>' if badge else ""
    st.markdown(f"""
    <div class="hero">
      {badge_html}
      <div class="hero-eyebrow">{eyebrow}</div>
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def section(label, title, desc=""):
    st.markdown(f"""
    <div class="section-label">{label}</div>
    <div class="section-title">{title}</div>
    {"<div class='section-desc'>" + desc + "</div>" if desc else ""}
    """, unsafe_allow_html=True)

def callout(text, kind="info", title=""):
    title_html = f'<div class="callout-title">{title}</div>' if title else ""
    st.markdown(f'<div class="callout callout-{kind}">{title_html}{text}</div>', unsafe_allow_html=True)

def metric_row(items):
    """items = list of (value, label)"""
    pills = "".join(f'<div class="metric-pill"><div class="val">{v}</div><div class="lbl">{l}</div></div>' for v, l in items)
    st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)

def progress_steps(steps, current):
    html = "<div class='steps'>"
    for i, s in enumerate(steps):
        cls = "done" if i < current else ("active" if i == current else "")
        html += f'<div class="step {cls}"><span class="step-num">0{i+1}</span>{s}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def energy_bar(val, max_val, label="Energy"):
    pct = min(max(val / max_val * 100, 0), 100)
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#7ab8d4;margin-bottom:4px;">
        <span>{label}</span>
        <span style="font-family:'Syne Mono',monospace;color:#00d4ff;">{val:.3f}</span>
      </div>
      <div class="energy-bar-wrap">
        <div class="energy-bar-fill" style="width:{pct}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_grid(data_df, key_prefix="nrn"):
    """Render 7×7 clickable neuron grid. Returns True if any cell changed."""
    changed = False
    for r in range(7):
        cols = st.columns(7)
        for c in range(7):
            idx = r * 7 + c
            is_on = bool(data_df.iloc[r, c])
            wrap_cls = "neuron-on" if is_on else ""
            with cols[c]:
                st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
                if st.button("·", key=f"{key_prefix}_{r}_{c}"):
                    data_df.iloc[r, c] = not is_on
                    changed = True
                st.markdown("</div>", unsafe_allow_html=True)
    return changed

def log_terminal(logs):
    lines = "".join(f'<p class="log-line">{l}</p>' for l in logs[-20:])
    st.markdown(f'<div class="log-terminal">{lines}</div>', unsafe_allow_html=True)

def concept_card_grid(items):
    """items = list of (icon, name, desc)"""
    html = '<div class="concept-grid">'
    for icon, name, desc in items:
        html += f"""
        <div class="concept-card">
          <span class="concept-icon">{icon}</span>
          <div class="concept-name">{name}</div>
          <div class="concept-desc">{desc}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    if "memories" not in st.session_state:
        st.session_state.memories = dict(DEFAULT_PATTERNS)
    if "net" not in st.session_state:
        net = HopfieldNetwork(49)
        net.train(list(st.session_state.memories.values()))
        st.session_state.net = net
    if "grid" not in st.session_state:
        st.session_state.grid = pd.DataFrame(np.zeros((7, 7), dtype=bool))
    if "narrative" not in st.session_state:
        st.session_state.narrative = "Awaiting input signal…"
    if "energy_hist" not in st.session_state:
        st.session_state.energy_hist = []
    if "logs" not in st.session_state:
        st.session_state.logs = ["[BOOT] Neural Engine v2.0 initialised.", f"[INFO] {len(st.session_state.memories)} patterns loaded into memory."]
    if "overlap_hist" not in st.session_state:
        st.session_state.overlap_hist = {k: [] for k in st.session_state.memories}
    if "module" not in st.session_state:
        st.session_state.module = "🏠 Introduction"

def flat_state():
    return np.where(st.session_state.grid.values.flatten(), 1.0, -1.0)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0: INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
def page_intro():
    hero(
        "Hopfield Network",
        "An interactive journey into associative memory — how a simple network of artificial neurons can store, recall, and reconstruct patterns from noise.",
        eyebrow="Neural Memory Lab · Learning Module",
        badge="Beginner Friendly"
    )

    c1, c2 = st.columns([1.6, 1])
    with c1:
        section("00 · The Big Idea", "What is a Hopfield Network?")
        st.markdown("""
        Imagine your brain struggling to remember a friend's face. Even from a blurry photo, 
        you instantly recognise them. That's **associative memory** — the ability to retrieve a 
        complete memory from a partial or noisy cue.

        A **Hopfield Network** (proposed by John Hopfield in 1982) is an artificial neural network 
        that works exactly this way. It's a network of neurons that:
        
        1. **Learns** patterns by adjusting the strength of connections between neurons.
        2. **Recalls** stored patterns when given a corrupted or incomplete version.
        3. **Converges** to a stable state — like a ball rolling into a valley.
        """)

        callout(
            "💡 This is one of the first neural networks that could be fully explained with physics "
            "and mathematics. It bridged neuroscience, physics, and computer science.",
            kind="tip", title="Why does it matter?"
        )

        section("01 · Key Concepts", "Three ideas power the whole system")
        concept_card_grid([
            ("⚡", "Neurons", "Binary units that are either ON (+1) or OFF (−1). Like pixels in a black-and-white image."),
            ("🔗", "Synaptic Weights", "Numbers on connections between neurons. Positive = neurons agree. Negative = neurons disagree."),
            ("🏔️", "Energy", "A measure of how 'stable' the current state is. The network always tries to minimise energy."),
        ])

    with c2:
        section("02 · Real-World Analogies", "")
        for icon, title, body in [
            ("🧲", "Magnetic Memory", "Like magnetic domains in a hard drive, neurons 'snap' into stable configurations."),
            ("📸", "Photo Restoration", "Given a scratched photograph, the network fills in the missing parts from memory."),
            ("🌊", "Ball in a Valley", "Energy minimisation = a ball rolling downhill until it settles in the deepest valley."),
            ("🔒", "Content-Addressable", "Unlike RAM (address → data), this is data → data. The pattern IS the address."),
        ]:
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.2rem; margin-bottom:0.7rem;">
              <span style="font-size:1.4rem;">{icon}</span>
              <span style="font-family:'Syne',sans-serif;font-weight:600;color:#e2f4ff;margin-left:0.5rem;">{title}</span>
              <div style="font-size:0.82rem;color:#7ab8d4;margin-top:0.4rem;line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    section("03 · Your Learning Path", "Five modules from zero to expert")
    progress_steps(["Introduction", "Neurons & States", "Hebbian Learning", "Energy & Attractors", "Playground"], 0)
    callout("Use the sidebar to navigate between modules. Each module builds on the last. Start with Module 1 → Neurons.", kind="info", title="How to use this app")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: NEURONS & BINARY STATES
# ─────────────────────────────────────────────────────────────────────────────
def page_neurons():
    hero(
        "Neurons & Binary States",
        "Every complex behaviour in the network comes from the simplest possible unit: a neuron that is either ON or OFF.",
        eyebrow="Module 01 · Foundations",
        badge="Step 1 of 4"
    )
    progress_steps(["Introduction", "Neurons & States", "Hebbian Learning", "Energy & Attractors", "Playground"], 1)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("What is a Neuron?", "The atom of the Hopfield Network")
        st.markdown("""
        In a Hopfield network, each neuron is a **binary unit** — it can only be in one of two states:

        | State | Value | Meaning |
        |-------|-------|---------|
        | **Firing** | `+1` | Neuron is **active** (bright pixel) |
        | **Silent** | `−1` | Neuron is **inactive** (dark pixel) |

        For a 7×7 grid, we have **49 neurons**. Their combined state — all 49 values — forms one 
        "snapshot" of the network, which we use to encode a pattern.
        """)

        callout(
            "Why +1 and −1 instead of 1 and 0? The math is much cleaner! "
            "The energy formula and Hebbian learning both simplify beautifully with bipolar values.",
            kind="tip", title="Mathematical note"
        )

        section("Update Rule", "How does a single neuron decide its next state?")
        st.markdown(r"""
        Each neuron $i$ looks at all other neurons and computes a **weighted sum** of their states:

        $$h_i = \sum_{j \neq i} W_{ij} \cdot s_j$$

        Then it applies a simple threshold:
        
        $$s_i^{new} = \begin{cases} +1 & \text{if } h_i \geq 0 \\ -1 & \text{if } h_i < 0 \end{cases}$$

        Think of it as a **vote**: connected neurons vote on whether neuron $i$ should be ON or OFF,
        weighted by the strength of their connection.
        """)

    with c2:
        section("Interactive Demo", "Click neurons to toggle them ON/OFF")
        callout("Click any cell below to toggle its state. Blue = ON (+1), Dark = OFF (−1)", kind="info")

        demo_grid = st.session_state.get("demo_grid", pd.DataFrame(np.zeros((7,7), dtype=bool)))
        changed = render_grid(demo_grid, key_prefix="demo")
        if changed:
            st.session_state.demo_grid = demo_grid
            st.rerun()

        on_count = int(demo_grid.values.sum())
        off_count = 49 - on_count
        metric_row([
            (on_count, "+1 Active"),
            (off_count, "−1 Silent"),
            (f"{on_count/49*100:.0f}%", "Density"),
        ])

        st.markdown("""
        <div class="card" style="margin-top:1rem;">
          <div style="font-family:'Syne Mono',monospace;font-size:0.75rem;color:#4fa8c8;">
            CURRENT STATE VECTOR (first 10 values)
          </div>
        """, unsafe_allow_html=True)
        flat = np.where(demo_grid.values.flatten(), 1, -1)
        state_str = " ".join([f"<span style='color:{'#00d4ff' if v==1 else '#3a6680'}'>{v:+d}</span>" for v in flat[:10]])
        st.markdown(f'<div style="font-family:\'Syne Mono\',monospace;font-size:0.9rem;margin-top:0.5rem;letter-spacing:0.05em;">{state_str} …</div></div>', unsafe_allow_html=True)

    st.divider()
    section("Quiz Check ✓", "Test your understanding")
    with st.expander("What does a neuron with value +1 represent?"):
        st.markdown("✅ **An active/firing neuron** — equivalent to a bright pixel in the pattern. In real brains, this corresponds to a neuron that is currently sending signals.")
    with st.expander("Why use +1/−1 instead of 1/0?"):
        st.markdown("✅ **Symmetry and cleaner math.** With bipolar values, the Hebbian learning rule (W = Σ pᵢ·pᵢᵀ) works elegantly. With 0/1 encoding, you'd need bias terms.")
    with st.expander("How many possible states can a 7×7 network have?"):
        st.markdown("✅ **2⁴⁹ ≈ 562 trillion states!** The network can only store ~7 stable patterns reliably (≈ 0.15 × N), but must navigate this vast state space.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: HEBBIAN LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def page_hebbian():
    hero(
        "Hebbian Learning",
        "\"Neurons that fire together, wire together.\" — Donald Hebb, 1949. The simplest and most powerful learning rule in neuroscience.",
        eyebrow="Module 02 · Learning Rule",
        badge="Step 2 of 4"
    )
    progress_steps(["Introduction", "Neurons & States", "Hebbian Learning", "Energy & Attractors", "Playground"], 2)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("The Learning Rule", "How patterns are etched into weights")
        st.markdown(r"""
        When we **train** the network on a pattern $\mathbf{p}$, we update the weight matrix:

        $$W \leftarrow W + \frac{1}{N} \mathbf{p} \cdot \mathbf{p}^T$$

        For multiple patterns $\{\mathbf{p}^1, \mathbf{p}^2, \ldots, \mathbf{p}^M\}$:

        $$W_{ij} = \frac{1}{N} \sum_{\mu=1}^{M} p_i^\mu \cdot p_j^\mu \quad (i \neq j)$$

        **What does this mean intuitively?**
        - If neurons $i$ and $j$ are **both ON** ($+1 \times +1 = +1$) → strengthen connection
        - If neurons $i$ and $j$ **disagree** ($+1 \times -1 = -1$) → weaken connection  
        - **Diagonal** $W_{ii} = 0$ — neurons don't connect to themselves
        """)

        callout(
            "This is a one-shot learning rule! You don't need gradient descent or backpropagation. "
            "Each pattern is stored in a single matrix addition. "
            "The network 'memorises' by superimposing outer products.",
            kind="tip", title="No backpropagation needed"
        )

        callout(
            "⚠️ Capacity Limit: A network with N neurons can reliably store about 0.138 × N patterns. "
            "For our 49-neuron network, that's only ~7 patterns. Beyond this, memories interfere with each other.",
            kind="warn", title="Storage capacity"
        )

    with c2:
        section("Live Weight Visualiser", "See how weights change as you add patterns")
        pat_to_show = st.multiselect(
            "Select patterns to train on:",
            list(DEFAULT_PATTERNS.keys()),
            default=list(DEFAULT_PATTERNS.keys())[:2]
        )

        if pat_to_show:
            temp_net = HopfieldNetwork(49)
            temp_net.train([DEFAULT_PATTERNS[k] for k in pat_to_show])

            fig = go.Figure(data=go.Heatmap(
                z=temp_net.W,
                colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[0.6,"#003355"],[0.75,"#0077aa"],[1,"#00d4ff"]],
                zmid=0,
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="#7ab8d4", size=9),
                    thickness=10,
                )
            ))
            style_fig(fig, h=340, title=f"Weight Matrix W — {len(pat_to_show)} pattern(s) trained")
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            w = temp_net.W
            metric_row([
                (f"{w.max():.3f}", "Max Weight"),
                (f"{w.min():.3f}", "Min Weight"),
                (f"{np.abs(w).mean():.3f}", "Avg |W|"),
                (f"{(w!=0).sum()}", "Active Synps"),
            ])
        else:
            st.info("Select at least one pattern above to visualise the weight matrix.")

    st.divider()
    section("Step-by-Step Walkthrough", "How a single pattern modifies the weight matrix")

    steps_demo = st.tabs(["Step 1: Choose Pattern", "Step 2: Compute Outer Product", "Step 3: Add to W"])
    with steps_demo[0]:
        p = DEFAULT_PATTERNS["Letter N"]
        fig = go.Figure(data=go.Heatmap(
            z=p.reshape(7,7),
            colorscale=[[0,"#0a1628"],[1,"#00d4ff"]],
            showscale=False
        ))
        style_fig(fig, h=200, title="Pattern: Letter N")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("We flatten this 7×7 grid into a 49-element vector **p** with values +1 or −1.")

    with steps_demo[1]:
        p = DEFAULT_PATTERNS["Letter N"]
        outer = np.outer(p, p)
        fig = go.Figure(data=go.Heatmap(
            z=outer, zmid=0,
            colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]],
            showscale=False
        ))
        style_fig(fig, h=200, title="Outer Product: p × pᵀ (49×49)")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("The outer product **p·pᵀ** creates a 49×49 matrix. Each cell = pᵢ × pⱼ.")

    with steps_demo[2]:
        p = DEFAULT_PATTERNS["Letter N"]
        w_final = np.outer(p, p) / 49
        np.fill_diagonal(w_final, 0)
        fig = go.Figure(data=go.Heatmap(
            z=w_final, zmid=0,
            colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]],
            showscale=False
        ))
        style_fig(fig, h=200, title="Normalised W after storing 'Letter N'")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Divide by N (=49) and zero the diagonal. This is your updated weight matrix!")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: ENERGY & ATTRACTORS
# ─────────────────────────────────────────────────────────────────────────────
def page_energy():
    hero(
        "Energy & Attractors",
        "The Hopfield network is a physical system. Every state has an energy level, and the network always \"rolls downhill\" toward stored memories.",
        eyebrow="Module 03 · Physics of Memory",
        badge="Step 3 of 4"
    )
    progress_steps(["Introduction", "Neurons & States", "Hebbian Learning", "Energy & Attractors", "Playground"], 3)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("The Energy Function", "Borrowed from statistical physics")
        st.markdown(r"""
        The **energy** of a network state $\mathbf{s}$ is defined as:

        $$E(\mathbf{s}) = -\frac{1}{2} \sum_{i \neq j} W_{ij} \cdot s_i \cdot s_j = -\frac{1}{2} \mathbf{s}^T \mathbf{W} \mathbf{s}$$

        **Key insights:**
        - **Lower energy = more stable state** (a memory attractor)
        - Stored patterns are **local energy minima** — valleys in the landscape
        - Every asynchronous update is **guaranteed to decrease or maintain energy**
        - The network monotonically converges — it can never oscillate indefinitely!
        """)

        section("Attractors", "What are they?")
        st.markdown("""
        An **attractor** is a state the network naturally gravitates toward.
        
        | Type | Description |
        |------|-------------|
        | **Stored memory** | A pattern you explicitly trained. Lowest energy wells. |
        | **Spurious state** | A mixture of stored patterns. Unintended but stable. |
        | **Reverse memory** | The *negation* of a stored pattern (also a stable state!). |
        """)

        callout(
            "🔍 Spurious states are like 'false memories'. If you store patterns A, B, C — "
            "the network may also create stable states that are weighted mixtures of those patterns. "
            "This is a fundamental limitation.",
            kind="warn", title="Spurious attractors"
        )

    with c2:
        section("Energy Landscape Simulation", "Explore a simplified 1D landscape")

        n_patterns = st.slider("Number of patterns stored", 1, 7, 3)
        noise_level = st.slider("Distance from nearest memory", 0, 25, 10)

        # Simulate energy across a 1D projection
        nets = HopfieldNetwork(49)
        chosen = list(DEFAULT_PATTERNS.values())[:n_patterns]
        nets.train(chosen)

        baseline = chosen[0].copy()
        corrupt = baseline.copy()
        flip_idx = np.random.choice(49, noise_level, replace=False)
        for i in flip_idx: corrupt[i] *= -1

        # Interpolate from corrupt to memory for landscape
        energies = []
        t_vals = np.linspace(0, 1, 40)
        for t in t_vals:
            interp = np.where(np.random.random(49) < t, baseline, corrupt).astype(float)
            energies.append(nets.energy(interp))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_vals, y=energies, mode='lines',
            line=dict(color='#00d4ff', width=3),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.06)',
            name='Energy'
        ))
        min_idx = int(np.argmin(energies))
        fig.add_trace(go.Scatter(
            x=[t_vals[min_idx]], y=[energies[min_idx]],
            mode='markers', marker=dict(color='#00ffcc', size=12, symbol='diamond'),
            name='Attractor'
        ))
        fig.add_annotation(x=t_vals[min_idx], y=energies[min_idx],
            text="Memory Attractor", showarrow=True, arrowcolor="#00ffcc",
            font=dict(color="#00ffcc", size=10), arrowhead=2)
        style_fig(fig, h=280, title="Energy Landscape (1D projection)")
        fig.update_xaxes(title="← Corrupted    Progress    Memory →")
        fig.update_yaxes(title="Energy E(s)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card" style="padding:1rem;">
          <div style="font-size:0.8rem;color:#7ab8d4;">
            <b style="color:#00d4ff;">{n_patterns}</b> patterns stored &nbsp;|&nbsp; 
            <b style="color:#00ffcc;">~{int(n_patterns/0.138/49*100)}%</b> of capacity used &nbsp;|&nbsp;
            Capacity limit: <b style="color:#ffb347;">{int(0.138*49)}</b> patterns
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    section("Convergence Proof (Intuition)", "Why the network always stabilises")
    col1, col2, col3 = st.columns(3)
    for col, title, body, icon in [
        (col1, "Asynchronous Updates", "One neuron updates at a time. Each update can only lower or maintain energy — never increase it.", "📉"),
        (col2, "Bounded Below", "Energy has a minimum value (−½ Σ|Wᵢⱼ|). It can't drop forever — it must stop somewhere.", "🔒"),
        (col3, "Finite States", "With N binary neurons, there are only 2ᴺ possible states. The network must reach a fixed point.", "🎯"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card card-glow" style="text-align:center;padding:1.2rem;">
              <span style="font-size:2rem;">{icon}</span>
              <div style="font-family:'Syne',sans-serif;font-weight:600;color:#e2f4ff;margin:0.5rem 0 0.4rem;">{title}</div>
              <div style="font-size:0.8rem;color:#7ab8d4;line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: FULL PLAYGROUND
# ─────────────────────────────────────────────────────────────────────────────
def page_playground():
    hero(
        "Neural Playground",
        "Put everything together. Store patterns, corrupt them with noise, and watch the Hopfield network reconstruct memories from chaos.",
        eyebrow="Module 04 · Experimentation Lab",
        badge="Full Playground"
    )
    progress_steps(["Introduction", "Neurons & States", "Hebbian Learning", "Energy & Attractors", "Playground"], 4)

    # ─── Control Hub ───
    with st.container():
        st.markdown('<div class="card card-glow" style="padding:1.2rem;">', unsafe_allow_html=True)
        hub_c1, hub_c2, hub_c3, hub_c4 = st.columns([1.2, 1.1, 1.1, 1])

        with hub_c1:
            st.markdown('<div class="section-label">Pattern Library</div>', unsafe_allow_html=True)
            p_keys = list(st.session_state.memories.keys())
            selected = st.selectbox("Pattern", ["— Free Draw —"] + p_keys, label_visibility="collapsed")
            if selected != "— Free Draw —":
                if st.button("📥 Load to Grid", use_container_width=True, type="primary"):
                    arr = st.session_state.memories[selected]
                    st.session_state.grid = pd.DataFrame((arr == 1).reshape(7, 7))
                    st.session_state.narrative = f"Pattern '{selected}' loaded. Ready to corrupt or recover."
                    st.session_state.logs.append(f"[LOAD] Pattern '{selected}' loaded to grid.")
                    st.rerun()

        with hub_c2:
            st.markdown('<div class="section-label">Noise Injection</div>', unsafe_allow_html=True)
            noise = st.slider("Corrupt bits", 1, 30, 10, label_visibility="collapsed")
            if st.button("🔴 Inject Noise", use_container_width=True):
                flat = st.session_state.grid.values.flatten().copy()
                idx = np.random.choice(49, noise, replace=False)
                for i in idx: flat[i] = not flat[i]
                st.session_state.grid = pd.DataFrame(flat.reshape(7, 7))
                st.session_state.narrative = f"⚡ {noise} neurons randomly flipped. Can the network recover?"
                st.session_state.logs.append(f"[NOISE] {noise} bits flipped.")
                st.rerun()

        with hub_c3:
            st.markdown('<div class="section-label">Teach New Pattern</div>', unsafe_allow_html=True)
            new_name = st.text_input("Pattern name", placeholder="e.g. My Letter", label_visibility="collapsed")
            if st.button("💾 Store Pattern", use_container_width=True, type="primary"):
                if new_name.strip():
                    flat = st.session_state.grid.values.flatten()
                    st.session_state.memories[new_name] = np.where(flat, 1.0, -1.0)
                    st.session_state.net.train(list(st.session_state.memories.values()))
                    st.session_state.overlap_hist[new_name] = []
                    st.session_state.narrative = f"Hebbian learning complete. '{new_name}' etched into weights."
                    st.session_state.logs.append(f"[LEARN] '{new_name}' stored. Total: {len(st.session_state.memories)} patterns.")
                    st.success(f"Stored '{new_name}'!")
                    st.rerun()
                else:
                    st.error("Give the pattern a name first!")

        with hub_c4:
            st.markdown('<div class="section-label">Danger Zone</div>', unsafe_allow_html=True)
            if st.button("🗑 Clear Grid", use_container_width=True):
                st.session_state.grid = pd.DataFrame(np.zeros((7, 7), dtype=bool))
                st.session_state.logs.append("[CLEAR] Grid wiped.")
                st.rerun()
            if st.button("💥 Reset All", use_container_width=True):
                st.session_state.memories = dict(DEFAULT_PATTERNS)
                st.session_state.net.train(list(st.session_state.memories.values()))
                st.session_state.grid = pd.DataFrame(np.zeros((7, 7), dtype=bool))
                st.session_state.energy_hist = []
                st.session_state.overlap_hist = {k: [] for k in st.session_state.memories}
                st.session_state.logs = ["[RESET] System reset to defaults."]
                st.session_state.narrative = "System reset. All default patterns reloaded."
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Main Area ───
    left, right = st.columns([1.05, 1])

    with left:
        section("Active Network Grid", "Click neurons to toggle · Blue = ON (+1) · Dark = OFF (−1)")
        changed = render_grid(st.session_state.grid, key_prefix="play")
        if changed:
            flat_b = flat_state()
            idx = np.unravel_index(
                next(i for i,(a,b) in enumerate(zip(
                    st.session_state.grid.values.flatten(),
                    ~st.session_state.grid.values.flatten()
                )) if True), (7,7)
            )
            st.session_state.narrative = "Manual edit detected. Grid updated."
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        rec1, rec2 = st.columns(2)
        with rec1:
            if st.button("▶ Full Recovery (40 steps)", use_container_width=True, type="primary"):
                curr = flat_state()
                st.session_state.energy_hist = []
                for i in range(40):
                    curr, upd, act = st.session_state.net.update_async(curr)
                    st.session_state.energy_hist.append(st.session_state.net.energy(curr))
                    for k, mem in st.session_state.memories.items():
                        if k not in st.session_state.overlap_hist:
                            st.session_state.overlap_hist[k] = []
                        st.session_state.overlap_hist[k].append(st.session_state.net.overlap(curr, mem))
                st.session_state.grid = pd.DataFrame((curr == 1).reshape(7, 7))
                st.session_state.narrative = f"Recovery complete after 40 steps. Energy: {st.session_state.energy_hist[-1]:.3f}"
                st.session_state.logs.append(f"[RUN] 40-step recovery done. ΔE = {st.session_state.energy_hist[-1]:.3f}")
                st.rerun()

        with rec2:
            if st.button("⏭ Single Step", use_container_width=True):
                curr = flat_state()
                curr, upd, act = st.session_state.net.update_async(curr)
                st.session_state.grid = pd.DataFrame((curr == 1).reshape(7, 7))
                e = st.session_state.net.energy(curr)
                st.session_state.energy_hist.append(e)
                st.session_state.narrative = f"Neuron #{upd} updated. Activation = {act:.3f}"
                st.session_state.logs.append(f"[STEP] Neuron #{upd} → {'ON' if curr[upd]==1 else 'OFF'}, act={act:.2f}")
                st.rerun()

    with right:
        section("Network Status", "Real-time metrics and dynamics")

        # Narrative box
        st.markdown(f"""
        <div class="card pulse" style="min-height:70px;display:flex;align-items:center;padding:1.2rem 1.5rem;margin-bottom:1rem;">
          <span style="font-family:'Syne Mono',monospace;font-size:0.92rem;color:#60b8d4;line-height:1.6;">
            {st.session_state.narrative}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Energy & Stats
        curr = flat_state()
        e = st.session_state.net.energy(curr)
        on_n = int((curr == 1).sum())
        energy_bar(abs(e), 15, "Network Energy |E|")

        metric_row([
            (f"{e:.2f}", "Energy E"),
            (on_n, "+1 Neurons"),
            (49 - on_n, "−1 Neurons"),
            (len(st.session_state.memories), "Memories"),
        ])

        # Pattern overlap radar
        st.markdown('<div class="section-label" style="margin-top:1rem;">Pattern Overlap (Similarity to Memories)</div>', unsafe_allow_html=True)
        overlaps = {k: st.session_state.net.overlap(curr, v) for k, v in st.session_state.memories.items()}
        if overlaps:
            keys = list(overlaps.keys())[:6]
            vals = [overlaps[k] for k in keys]
            fig = go.Figure()
            colors = ['#00d4ff' if v > 0.6 else '#3a6680' if v < 0 else '#0077aa' for v in vals]
            fig.add_trace(go.Bar(
                x=keys, y=vals,
                marker_color=colors,
                text=[f"{v:.2f}" for v in vals],
                textposition='outside',
                textfont=dict(color='#7ab8d4', size=10, family='Syne Mono'),
            ))
            fig.add_hline(y=0.9, line_dash="dot", line_color="#00ffcc", annotation_text="Recall Threshold",
                         annotation_font=dict(color="#00ffcc", size=9))
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.1)")
            style_fig(fig, h=220, title="")
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
            fig.update_yaxes(range=[-1.1, 1.3])
            st.plotly_chart(fig, use_container_width=True)

    # ─── Energy History & Diagnostics ───
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Energy Descent", "🕸 Weight Matrix", "📋 Activity Log", "📚 Stored Patterns"])

    with tab1:
        if st.session_state.energy_hist:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.energy_hist,
                mode='lines+markers',
                line=dict(color='#00d4ff', width=2.5),
                marker=dict(size=4, color='#00ffcc'),
                fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
                name='Energy'
            ))
            style_fig(fig, h=260, title="Energy Landscape Descent During Recovery")
            fig.update_xaxes(title="Update Steps")
            fig.update_yaxes(title="Energy E(s)")
            st.plotly_chart(fig, use_container_width=True)
            callout(
                f"Energy started at <b>{st.session_state.energy_hist[0]:.3f}</b> and ended at "
                f"<b>{st.session_state.energy_hist[-1]:.3f}</b> — a decrease of "
                f"<b>{abs(st.session_state.energy_hist[-1] - st.session_state.energy_hist[0]):.3f}</b>. "
                "Lower energy = closer to a stored memory attractor.",
                kind="info", title="What just happened?"
            )
        else:
            st.info("Run a Full Recovery to see the energy descent curve here.")

    with tab2:
        section("Synaptic Weight Map", "Every stored pattern leaves its mark on this 49×49 matrix")
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.net.W,
            colorscale=[[0,"#0a0e1a"],[0.35,"#002244"],[0.5,"#001133"],[0.65,"#004488"],[1,"#00d4ff"]],
            zmid=0, showscale=True,
            colorbar=dict(tickfont=dict(color="#7ab8d4", size=9), thickness=12)
        ))
        style_fig(fig, h=420, title=f"W — {len(st.session_state.memories)} patterns encoded")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("Activity Log", "Full history of network operations")
        log_terminal(st.session_state.logs)

    with tab4:
        section("Stored Pattern Library", "Visual overview of all memorised patterns")
        keys = list(st.session_state.memories.keys())
        chunk = 5
        for row_start in range(0, len(keys), chunk):
            cols = st.columns(chunk)
            for ci, k in enumerate(keys[row_start:row_start+chunk]):
                pat = st.session_state.memories[k]
                fig = go.Figure(data=go.Heatmap(
                    z=pat.reshape(7, 7),
                    colorscale=[[0,"#0a1628"],[1,"#00d4ff"]],
                    showscale=False, zmin=-1, zmax=1
                ))
                fig.update_layout(
                    **PLOT_LAYOUT, height=130,
                    title=dict(text=k, font=dict(size=10, color="#7ab8d4")),
                    margin=dict(l=2, r=2, t=24, b=2)
                )
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                cols[ci].plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          🧠 Hopfield<span>Lab</span>
        </div>
        """, unsafe_allow_html=True)

        module = st.radio(
            "Navigation",
            ["🏠 Introduction", "⚡ Neurons & States", "🔗 Hebbian Learning", "🌋 Energy & Attractors", "🎮 Playground"],
            label_visibility="collapsed"
        )
        st.session_state.module = module

        st.markdown("---")
        st.markdown('<div class="section-label">Network Status</div>', unsafe_allow_html=True)
        n_mem = len(st.session_state.memories)
        capacity = int(0.138 * 49)
        pct = n_mem / capacity * 100
        st.markdown(f"""
        <div style="margin: 0.5rem 0 0.3rem; font-size:0.8rem; color:#7ab8d4;">
          Stored Patterns: <span style="color:#00d4ff;font-family:'Syne Mono',monospace;">{n_mem}</span>
          / <span style="color:#7ab8d4;">{capacity}</span> capacity
        </div>
        <div class="energy-bar-wrap">
          <div class="energy-bar-fill" style="width:{min(pct,100):.0f}%;background:{'linear-gradient(90deg,#ff5580,#ffb347)' if pct>80 else 'linear-gradient(90deg,#00a8cc,#00d4ff,#00ffcc)'};"></div>
        </div>
        """, unsafe_allow_html=True)

        if n_mem >= capacity:
            st.warning("⚠️ Near capacity — spurious states likely!")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#3a6680;line-height:1.8;">
          <div style="color:#4fa8c8;margin-bottom:0.3rem;font-family:'Syne Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;">QUICK REFERENCE</div>
          <b style="color:#7ab8d4;">N</b> = 49 neurons<br>
          <b style="color:#7ab8d4;">Capacity</b> ≈ 0.138N = 7<br>
          <b style="color:#7ab8d4;">Update rule</b>: sign(Ws)<br>
          <b style="color:#7ab8d4;">Energy</b>: −½ sᵀWs<br>
          <b style="color:#7ab8d4;">Learning</b>: W += ppᵀ/N
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.7rem;color:#3a6680;text-align:center;line-height:1.6;">
          Based on Hopfield (1982)<br>
          <i>Neural Networks and Physical Systems with Emergent Collective Computational Abilities</i>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    init_state()
    sidebar()

    m = st.session_state.module
    if m == "🏠 Introduction":
        page_intro()
    elif m == "⚡ Neurons & States":
        page_neurons()
    elif m == "🔗 Hebbian Learning":
        page_hebbian()
    elif m == "🌋 Energy & Attractors":
        page_energy()
    elif m == "🎮 Playground":
        page_playground()

if __name__ == "__main__":
    main()