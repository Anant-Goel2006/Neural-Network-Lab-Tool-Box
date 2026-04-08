"""
Hopfield Network — Interactive Learning Module  
Full-featured: Draw & Recognize, AI Tutor, Energy Landscape, Playground
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time, math, os, json
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    st.set_page_config(page_title="Hopfield Network", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# NVIDIA AI TUTOR
# ═══════════════════════════════════════════════════════════════
def get_nvidia_key():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        try:
            key = st.secrets.get("NVIDIA_API_KEY", None)
        except Exception:
            pass
    return key

def ai_explain(prompt, context=""):
    key = get_nvidia_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        msgs = [
            {"role": "system", "content": "You are a brilliant neural network professor. Explain Hopfield network concepts clearly with analogies. Keep responses under 200 words. Use markdown formatting."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
        resp = client.chat.completions.create(model="meta/llama-3.1-8b-instruct", messages=msgs, temperature=0.7, max_tokens=400)
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI unavailable: {e}"

def ai_tutor_widget(default_q, context, key_suffix=""):
    with st.expander("🤖 Ask AI Tutor", expanded=False):
        q = st.text_input("Your question:", value=default_q, key=f"ai_q_{key_suffix}")
        if st.button("Ask AI", key=f"ai_btn_{key_suffix}", type="primary"):
            with st.spinner("🧠 Thinking..."):
                ans = ai_explain(q, context)
                if ans:
                    st.markdown(f'<div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:10px;padding:1.2rem;color:#e2f4ff;font-size:0.92rem;line-height:1.7;">{ans}</div>', unsafe_allow_html=True)
                else:
                    st.info("Add your NVIDIA API key to .env or Streamlit Secrets to enable AI Tutor.")

# ═══════════════════════════════════════════════════════════════
# HOPFIELD NETWORK ENGINE
# ═══════════════════════════════════════════════════════════════
class HopfieldNetwork:
    def __init__(self, size=100):
        self.N = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        self.W = np.zeros((self.N, self.N))
        if not patterns: return
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        return -0.5 * float(state @ self.W @ state)

    def update_async(self, state, idx=None):
        s = state.copy()
        if idx is None: idx = np.random.randint(0, self.N)
        act = float(self.W[idx] @ s)
        s[idx] = 1 if act >= 0 else -1
        return s, idx, act

    def update_sync(self, state):
        return np.where(self.W @ state >= 0, 1, -1).astype(float)

    def overlap(self, state, memory):
        return float(state @ memory) / self.N

    def recover(self, state, steps=50):
        s = state.copy()
        history = [s.copy()]
        energies = [self.energy(s)]
        for _ in range(steps):
            s, _, _ = self.update_async(s)
            history.append(s.copy())
            energies.append(self.energy(s))
        return s, history, energies

# ═══════════════════════════════════════════════════════════════
# 10x10 PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════
def b(grid):
    return np.array(grid).flatten().astype(float) * 2 - 1

GRID_SIZE = 10
DEFAULT_PATTERNS = {
    "A": b([[0,0,0,1,1,1,1,0,0,0],[0,0,1,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1]]),
    "B": b([[1,1,1,1,1,1,1,0,0,0],[1,0,0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1,0,0],[1,1,1,1,1,1,1,0,0,0],[1,0,0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1,0,0],[1,1,1,1,1,1,1,0,0,0]]),
    "C": b([[0,0,1,1,1,1,1,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,1,0],[0,0,1,1,1,1,1,1,0,0]]),
    "H": b([[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1]]),
    "L": b([[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1]]),
    "N": b([[1,0,0,0,0,0,0,0,0,1],[1,1,0,0,0,0,0,0,0,1],[1,0,1,0,0,0,0,0,0,1],[1,0,0,1,0,0,0,0,0,1],[1,0,0,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,0,0,1],[1,0,0,0,0,0,1,0,0,1],[1,0,0,0,0,0,0,1,0,1],[1,0,0,0,0,0,0,0,1,1],[1,0,0,0,0,0,0,0,0,1]]),
    "T": b([[1,1,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0]]),
    "X": b([[1,0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,1,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,1]]),
    "0": b([[0,0,1,1,1,1,1,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1,0],[0,0,1,1,1,1,1,1,0,0]]),
    "1": b([[0,0,0,0,1,1,0,0,0,0],[0,0,0,1,1,1,0,0,0,0],[0,0,1,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,1,1,1,1,1,1,1,1,0]]),
    "7": b([[1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]]),
    "Cross": b([[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0]]),
    "Diamond": b([[0,0,0,0,1,1,0,0,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,1,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,0,0,1,1,0,0,0,0]]),
}

# ═══════════════════════════════════════════════════════════════
# PLOTLY STYLING
# ═══════════════════════════════════════════════════════════════
PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,40,0.6)", font=dict(family="Space Grotesk", color="#7ab8d4"), margin=dict(l=10,r=10,t=30,b=10))

def style_fig(fig, h=250, title=""):
    fig.update_layout(**PLOT_BG, height=h, title=dict(text=title, font=dict(color="#e2f4ff", size=13)))
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)", tickfont=dict(size=10))

def pattern_heatmap(arr, title="", h=200):
    fig = go.Figure(data=go.Heatmap(z=arr.reshape(GRID_SIZE,GRID_SIZE), colorscale=[[0,"#0a1628"],[1,"#00d4ff"]], showscale=False, zmin=-1, zmax=1))
    style_fig(fig, h=h, title=title)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

# ═══════════════════════════════════════════════════════════════
# CSS THEME
# ═══════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne+Mono&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg-void:#020817;--bg-deep:#050f1f;--bg-card:rgba(10,22,40,0.85);--bg-card2:rgba(13,31,56,0.85);
    --border:rgba(0,212,255,0.15);--border-glow:rgba(0,212,255,0.4);
    --cyan:#00d4ff;--teal:#00ffcc;--amber:#ffb347;--rose:#ff5580;
    --text-bright:#e2f4ff;--text-mid:#7ab8d4;--text-dim:#3a6680;
    --mono:'Syne Mono',monospace;--display:'Syne',sans-serif;--body:'Space Grotesk',sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{background:transparent !important;font-family:var(--body);color:var(--text-bright);}
[data-testid="stSidebar"]{background:var(--bg-deep) !important;border-right:1px solid var(--border);}
[data-testid="stHeader"]{background:transparent !important;}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg-void)}::-webkit-scrollbar-thumb{background:var(--cyan-dim,#0099bb);border-radius:4px}
.hero{position:relative;padding:2.5rem 2rem 2rem;background:radial-gradient(ellipse at 20% 50%,rgba(0,212,255,0.08) 0%,transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(0,255,204,0.05) 0%,transparent 50%),var(--bg-card);border:1px solid var(--border);border-radius:20px;overflow:hidden;margin-bottom:2rem;}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),var(--teal),transparent);animation:scanline 3s ease-in-out infinite;}
@keyframes scanline{0%,100%{opacity:0.4}50%{opacity:1}}
.hero-eyebrow{font-family:var(--mono);font-size:0.72rem;letter-spacing:0.25em;color:var(--cyan);text-transform:uppercase;margin-bottom:0.75rem;opacity:0.8;}
.hero-title{font-family:var(--display);font-size:clamp(1.8rem,4vw,3rem);font-weight:800;line-height:1.1;background:linear-gradient(135deg,#e2f4ff 30%,var(--cyan) 70%,var(--teal) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem;}
.hero-sub{font-size:1rem;color:var(--text-mid);max-width:600px;line-height:1.7;}
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1.5rem;margin-bottom:1rem;position:relative;overflow:hidden;}
.card:hover{border-color:var(--border-glow);}
.card-glow{box-shadow:0 0 30px rgba(0,212,255,0.08),inset 0 1px 0 rgba(0,212,255,0.1);}
.section-label{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.2em;color:var(--cyan);text-transform:uppercase;margin-bottom:0.3rem;}
.section-title{font-family:var(--display);font-size:1.5rem;font-weight:700;color:var(--text-bright);margin-bottom:0.2rem;}
.section-desc{font-size:0.9rem;color:var(--text-mid);margin-bottom:1.2rem;}
.callout{padding:1.2rem 1.4rem;border-radius:10px;margin:1rem 0;border-left:3px solid;font-size:0.93rem;line-height:1.65;}
.callout-info{background:rgba(0,212,255,0.06);border-color:var(--cyan);color:var(--text-mid);}
.callout-tip{background:rgba(0,255,204,0.06);border-color:var(--teal);color:var(--text-mid);}
.callout-warn{background:rgba(255,179,71,0.06);border-color:var(--amber);color:#c8a070;}
.callout-title{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.4rem;opacity:0.9;}
.metric-row{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;}
.metric-pill{flex:1;min-width:100px;background:var(--bg-card2);border:1px solid var(--border);border-radius:10px;padding:1rem;text-align:center;}
.metric-pill .val{font-family:var(--mono);font-size:1.6rem;font-weight:600;color:var(--cyan);}
.metric-pill .lbl{font-size:0.72rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;}
.steps{display:flex;gap:0;margin:1.5rem 0;}.step{flex:1;text-align:center;padding:0.7rem 0.4rem;border-top:2px solid rgba(13,31,56,0.85);font-size:0.75rem;color:var(--text-dim);cursor:default;}
.step.active{border-color:var(--cyan);color:var(--cyan);font-weight:600;}.step.done{border-color:var(--teal);color:var(--teal);}
.step-num{display:block;font-family:var(--mono);font-size:0.6rem;opacity:0.6;margin-bottom:0.2rem;}
.badge{display:inline-block;padding:0.2em 0.7em;border-radius:99px;font-family:var(--mono);font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;}
.badge-cyan{background:rgba(0,212,255,0.12);color:var(--cyan);border:1px solid rgba(0,212,255,0.25);}
.energy-bar-wrap{background:var(--bg-card2);border-radius:99px;height:10px;overflow:hidden;margin:0.5rem 0;border:1px solid var(--border);}
.energy-bar-fill{height:100%;border-radius:99px;transition:width 0.5s ease;background:linear-gradient(90deg,#00a8cc,#00d4ff,#00ffcc);}
.log-terminal{background:#010a14;border:1px solid var(--border);border-radius:10px;padding:1rem;font-family:var(--mono);font-size:0.78rem;color:#4fa8c8;max-height:200px;overflow-y:auto;line-height:1.8;}
.log-line{margin:0;opacity:0.85;}.log-line:last-child{color:var(--cyan);opacity:1;}.log-line::before{content:'› ';color:var(--teal);}
.concept-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0;}
.concept-card{background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:1.2rem;text-align:center;transition:0.2s;}
.concept-card:hover{border-color:var(--border-glow);transform:translateY(-2px);}
.concept-icon{font-size:2rem;margin-bottom:0.6rem;display:block;}
.concept-name{font-family:var(--display);font-size:0.92rem;font-weight:600;color:var(--text-bright);margin-bottom:0.4rem;}
.concept-desc{font-size:0.78rem;color:var(--text-dim);line-height:1.5;}
.js-plotly-plot .plotly{background:transparent !important;}
.sidebar-brand{font-family:var(--display);font-weight:800;font-size:1.1rem;color:var(--cyan);padding:0.5rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;}
.sidebar-brand span{color:var(--teal);}
div.stButton > button[kind="primary"]{background:linear-gradient(135deg,#0077aa,#00a8cc) !important;border:none !important;color:white !important;font-weight:600 !important;box-shadow:0 4px 20px rgba(0,168,204,0.3) !important;border-radius:8px !important;height:38px !important;width:auto !important;padding:0 1.2rem !important;font-size:0.88rem !important;}
div.stButton > button[kind="primary"]:hover{box-shadow:0 4px 28px rgba(0,212,255,0.5) !important;}
[data-testid="stTabs"] button{font-family:var(--body) !important;font-size:0.88rem !important;color:var(--text-dim) !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--cyan) !important;border-bottom-color:var(--cyan) !important;}
hr{border-color:var(--border) !important;}
@keyframes pulse-glow{0%,100%{box-shadow:0 0 8px rgba(0,212,255,0.3);}50%{box-shadow:0 0 24px rgba(0,212,255,0.7);}}.pulse{animation:pulse-glow 2s ease-in-out infinite;}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{font-family:var(--display);color:var(--text-bright);}
.stMarkdown code{font-family:var(--mono);background:rgba(0,212,255,0.08);color:var(--cyan);border-radius:4px;padding:0.1em 0.35em;font-size:0.88em;border:1px solid var(--border);}
</style>
"""

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def hero(title, subtitle, eyebrow="", badge=""):
    bh = f'<span class="badge badge-cyan" style="margin-bottom:1rem;display:inline-block;">{badge}</span><br>' if badge else ""
    st.markdown(f'<div class="hero">{bh}<div class="hero-eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>', unsafe_allow_html=True)

def section(label, title, desc=""):
    d = f"<div class='section-desc'>{desc}</div>" if desc else ""
    st.markdown(f'<div class="section-label">{label}</div><div class="section-title">{title}</div>{d}', unsafe_allow_html=True)

def callout(text, kind="info", title=""):
    th = f'<div class="callout-title">{title}</div>' if title else ""
    st.markdown(f'<div class="callout callout-{kind}">{th}{text}</div>', unsafe_allow_html=True)

def metric_row(items):
    pills = "".join(f'<div class="metric-pill"><div class="val">{v}</div><div class="lbl">{l}</div></div>' for v,l in items)
    st.markdown(f'<div class="metric-row">{pills}</div>', unsafe_allow_html=True)

def progress_steps(steps, current):
    html = "<div class='steps'>"
    for i,s in enumerate(steps):
        cls = "done" if i < current else ("active" if i == current else "")
        html += f'<div class="step {cls}"><span class="step-num">0{i+1}</span>{s}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

def energy_bar(val, max_val, label="Energy"):
    pct = min(max(abs(val)/max_val*100,0),100)
    st.markdown(f'<div style="margin:0.5rem 0;"><div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#7ab8d4;margin-bottom:4px;"><span>{label}</span><span style="font-family:var(--mono);color:#00d4ff;">{val:.3f}</span></div><div class="energy-bar-wrap"><div class="energy-bar-fill" style="width:{pct}%;"></div></div></div>', unsafe_allow_html=True)

def log_terminal(logs):
    lines = "".join(f'<p class="log-line">{l}</p>' for l in logs[-20:])
    st.markdown(f'<div class="log-terminal">{lines}</div>', unsafe_allow_html=True)

def concept_card_grid(items):
    html = '<div class="concept-grid">'
    for icon,name,desc in items:
        html += f'<div class="concept-card"><span class="concept-icon">{icon}</span><div class="concept-name">{name}</div><div class="concept-desc">{desc}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

def render_grid(data_df, key_prefix="nrn"):
    changed = False
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c in range(GRID_SIZE):
            is_on = bool(data_df.iloc[r,c])
            with cols[c]:
                st.markdown(f'<div class="{"neuron-on" if is_on else ""}">', unsafe_allow_html=True)
                if st.button("·", key=f"{key_prefix}_{r}_{c}"):
                    data_df.iloc[r,c] = not is_on
                    changed = True
                st.markdown("</div>", unsafe_allow_html=True)
    return changed

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init_state():
    if "hop_memories" not in st.session_state:
        st.session_state.hop_memories = dict(DEFAULT_PATTERNS)
    if "hop_net" not in st.session_state:
        net = HopfieldNetwork(GRID_SIZE * GRID_SIZE)
        net.train(list(st.session_state.hop_memories.values()))
        st.session_state.hop_net = net
    if "hop_grid" not in st.session_state:
        st.session_state.hop_grid = pd.DataFrame(np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool))
    if "hop_narrative" not in st.session_state:
        st.session_state.hop_narrative = "Awaiting input signal…"
    if "hop_energy_hist" not in st.session_state:
        st.session_state.hop_energy_hist = []
    if "hop_logs" not in st.session_state:
        st.session_state.hop_logs = ["[BOOT] Hopfield Engine v3.0 initialised.", f"[INFO] {len(st.session_state.hop_memories)} patterns loaded."]
    if "hop_module" not in st.session_state:
        st.session_state.hop_module = "🏠 Introduction"

def flat_state():
    return np.where(st.session_state.hop_grid.values.flatten(), 1.0, -1.0)

# ═══════════════════════════════════════════════════════════════
# PAGE 0: INTRODUCTION
# ═══════════════════════════════════════════════════════════════
def page_intro():
    hero("Hopfield Network", "An interactive journey into associative memory — how neurons store, recall, and reconstruct patterns from noise.", eyebrow="Neural Memory Lab · Learning Module", badge="Beginner Friendly")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        section("00 · The Big Idea", "What is a Hopfield Network?")
        st.markdown("""
        Imagine struggling to remember a friend's face. Even from a blurry photo,
        you instantly recognise them. That's **associative memory**.

        A **Hopfield Network** (1982) is an artificial neural network that works this way:

        1. **Learns** patterns by adjusting connection strengths between neurons
        2. **Recalls** stored patterns when given a corrupted or incomplete version
        3. **Converges** to a stable state — like a ball rolling into a valley
        """)
        callout("💡 This is one of the first neural networks fully explainable with physics and mathematics. It bridged neuroscience, physics, and computer science.", kind="tip", title="Why does it matter?")

        section("01 · Key Concepts", "Three ideas power the whole system")
        concept_card_grid([
            ("⚡", "Neurons", "Binary units: ON (+1) or OFF (−1). Like pixels in a black-and-white image."),
            ("🔗", "Synaptic Weights", "Numbers on connections. Positive = agree. Negative = disagree."),
            ("🏔️", "Energy", "A stability measure. The network always tries to minimise energy."),
        ])

    with c2:
        section("02 · Real-World Analogies", "")
        for icon, title, body in [
            ("🧲", "Magnetic Memory", "Like magnetic domains, neurons 'snap' into stable configurations."),
            ("📸", "Photo Restoration", "Given a scratched photo, the network fills in missing parts from memory."),
            ("🌊", "Ball in a Valley", "Energy minimisation = a ball rolling downhill until it settles."),
            ("🔒", "Content-Addressable", "Unlike RAM (address→data), this is data→data. The pattern IS the address."),
        ]:
            st.markdown(f'<div class="card" style="padding:1rem 1.2rem;margin-bottom:0.7rem;"><span style="font-size:1.4rem;">{icon}</span><span style="font-family:var(--display);font-weight:600;color:#e2f4ff;margin-left:0.5rem;">{title}</span><div style="font-size:0.82rem;color:#7ab8d4;margin-top:0.4rem;line-height:1.5;">{body}</div></div>', unsafe_allow_html=True)

    st.divider()
    section("03 · Your Learning Path", "Six modules from zero to expert")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 0)
    callout("Use the sidebar to navigate between modules. Each builds on the last.", kind="info", title="How to use this app")
    ai_tutor_widget("What makes Hopfield networks different from other neural networks?", "Introduction to Hopfield networks", "intro")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: NEURONS & STATES
# ═══════════════════════════════════════════════════════════════
def page_neurons():
    hero("Neurons & Binary States", "Every complex behaviour comes from the simplest unit: a neuron that is either ON or OFF.", eyebrow="Module 01 · Foundations", badge="Step 1 of 5")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 1)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("What is a Neuron?", "The atom of the Hopfield Network")
        st.markdown("""
        Each neuron is a **binary unit** — only two states:

        | State | Value | Meaning |
        |-------|-------|---------|
        | **Firing** | `+1` | Active (bright pixel) |
        | **Silent** | `−1` | Inactive (dark pixel) |

        For a 10×10 grid, we have **100 neurons**. Their combined state
        encodes a pattern.
        """)
        callout("Why +1/−1 instead of 1/0? The math is much cleaner! Energy and Hebbian learning both simplify beautifully with bipolar values.", kind="tip", title="Mathematical note")

        section("Update Rule", "How does a neuron decide its next state?")
        st.markdown(r"""
        Each neuron $i$ computes a **weighted sum**:

        $$h_i = \sum_{j \neq i} W_{ij} \cdot s_j$$

        Then applies a threshold:
        $$s_i^{new} = \begin{cases} +1 & \text{if } h_i \geq 0 \\ -1 & \text{if } h_i < 0 \end{cases}$$

        Think of it as a **vote** — connected neurons vote whether neuron $i$ should be ON or OFF.
        """)

    with c2:
        section("Interactive Demo", "Click neurons to toggle ON/OFF")
        callout("Click any cell to toggle. Blue = ON (+1), Dark = OFF (−1)", kind="info")
        demo_key = "demo_grid_10"
        if demo_key not in st.session_state:
            st.session_state[demo_key] = pd.DataFrame(np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool))
        demo_grid = st.session_state[demo_key]
        changed = render_grid(demo_grid, key_prefix="demo10")
        if changed:
            st.session_state[demo_key] = demo_grid
            st.rerun()

        on_count = int(demo_grid.values.sum())
        metric_row([(on_count, "+1 Active"), (GRID_SIZE**2 - on_count, "−1 Silent"), (f"{on_count/GRID_SIZE**2*100:.0f}%", "Density")])

    st.divider()
    section("Quiz Check ✓", "Test your understanding")
    with st.expander("What does +1 represent?"):
        st.markdown("✅ **An active/firing neuron** — a bright pixel.")
    with st.expander("Why +1/−1 instead of 1/0?"):
        st.markdown("✅ **Symmetry and cleaner math.** Hebbian learning (W = Σ p·pᵀ) works elegantly.")
    with st.expander("How many states can a 10×10 network have?"):
        st.markdown("✅ **2¹⁰⁰ ≈ 1.27 × 10³⁰ states!** But only ~14 stable patterns (0.138 × N).")
    ai_tutor_widget("Explain the neuron update rule with a simple example", "Neurons & binary states in Hopfield networks", "neurons")

# ═══════════════════════════════════════════════════════════════
# PAGE 2: HEBBIAN LEARNING
# ═══════════════════════════════════════════════════════════════
def page_hebbian():
    hero("Hebbian Learning", '"Neurons that fire together, wire together." — Donald Hebb, 1949', eyebrow="Module 02 · Learning Rule", badge="Step 2 of 5")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 2)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("The Learning Rule", "How patterns are etched into weights")
        st.markdown(r"""
        When we **train** on a pattern $\mathbf{p}$, we update the weight matrix:
        $$W \leftarrow W + \frac{1}{N} \mathbf{p} \cdot \mathbf{p}^T$$

        For multiple patterns $\{\mathbf{p}^1, \ldots, \mathbf{p}^M\}$:
        $$W_{ij} = \frac{1}{N} \sum_{\mu=1}^{M} p_i^\mu \cdot p_j^\mu \quad (i \neq j)$$

        **Intuitively:**
        - Both ON ($+1 × +1 = +1$) → strengthen connection
        - Disagreeing ($+1 × -1 = -1$) → weaken connection
        - Diagonal $W_{ii} = 0$ — no self-connections
        """)
        callout("This is a one-shot learning rule! No gradient descent needed. Each pattern is stored in a single matrix addition.", kind="tip", title="No backpropagation needed")
        callout("⚠️ Capacity: ~0.138 × N patterns. For 100 neurons, that's ~14 patterns. Beyond this, memories interfere.", kind="warn", title="Storage capacity")

    with c2:
        section("Live Weight Visualiser", "See how weights change as you add patterns")
        pat_to_show = st.multiselect("Select patterns to train on:", list(DEFAULT_PATTERNS.keys()), default=list(DEFAULT_PATTERNS.keys())[:3])
        if pat_to_show:
            temp_net = HopfieldNetwork(GRID_SIZE**2)
            temp_net.train([DEFAULT_PATTERNS[k] for k in pat_to_show])
            fig = go.Figure(data=go.Heatmap(z=temp_net.W, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[0.75,"#0077aa"],[1,"#00d4ff"]], zmid=0, showscale=True, colorbar=dict(tickfont=dict(color="#7ab8d4",size=9),thickness=10)))
            style_fig(fig, h=340, title=f"Weight Matrix W — {len(pat_to_show)} pattern(s)")
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)
            w = temp_net.W
            metric_row([(f"{w.max():.3f}", "Max Weight"), (f"{w.min():.3f}", "Min Weight"), (f"{np.abs(w).mean():.3f}", "Avg |W|"), (f"{(w!=0).sum()}", "Active Synapses")])
        else:
            st.info("Select at least one pattern above.")

    st.divider()
    section("Step-by-Step Walkthrough", "How a single pattern modifies W")
    tabs = st.tabs(["Step 1: Choose Pattern", "Step 2: Outer Product", "Step 3: Add to W"])
    with tabs[0]:
        p = DEFAULT_PATTERNS["N"]
        st.plotly_chart(pattern_heatmap(p, "Pattern: Letter N", 200), use_container_width=True)
        st.markdown("Flatten this 10×10 grid into a 100-element vector **p** with values +1 or −1.")
    with tabs[1]:
        outer = np.outer(DEFAULT_PATTERNS["N"], DEFAULT_PATTERNS["N"])
        fig = go.Figure(data=go.Heatmap(z=outer, zmid=0, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]], showscale=False))
        style_fig(fig, h=250, title="Outer Product: p × pᵀ (100×100)")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        w_final = np.outer(DEFAULT_PATTERNS["N"], DEFAULT_PATTERNS["N"]) / (GRID_SIZE**2)
        np.fill_diagonal(w_final, 0)
        fig = go.Figure(data=go.Heatmap(z=w_final, zmid=0, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]], showscale=False))
        style_fig(fig, h=250, title="Normalised W after storing 'N'")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    ai_tutor_widget("Why is the diagonal of W always zero?", "Hebbian learning rule in Hopfield networks", "hebbian")

# ═══════════════════════════════════════════════════════════════
# PAGE 3: DRAW & RECOGNIZE (WHITEBOARD)
# ═══════════════════════════════════════════════════════════════
def page_draw():
    hero("Draw & Recognize", "Draw a character or number on the whiteboard. The Hopfield network will attempt to recall the closest stored pattern from memory.", eyebrow="Module 03 · Pattern Recognition", badge="Interactive Canvas")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 3)

    if not CANVAS_AVAILABLE:
        st.error("⚠️ `streamlit-drawable-canvas` is not installed. Run: `pip install streamlit-drawable-canvas`")
        return

    c1, c2 = st.columns([1.2, 1])
    with c1:
        section("✏️ Whiteboard", "Draw a letter or number (A-Z, 0-9)")
        callout("Draw boldly with thick strokes. The drawing will be downsampled to a 10×10 grid for recognition.", kind="info")

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=18,
            stroke_color="#00d4ff",
            background_color="#0a1628",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="hopfield_canvas",
        )

        col_a, col_b, col_c = st.columns(3)
        recognize_clicked = col_a.button("🧠 Recognize", type="primary", use_container_width=True)
        clear_clicked = col_b.button("🗑 Clear", use_container_width=True)

        if recognize_clicked and canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            gray = img.convert('L')
            small = gray.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
            arr = np.array(small)
            binary = np.where(arr > 80, 1.0, -1.0)
            noisy = binary.flatten()

            net = st.session_state.hop_net
            recovered, history, energies = net.recover(noisy, steps=60)

            overlaps = {k: net.overlap(recovered, v) for k, v in st.session_state.hop_memories.items()}
            best_match = max(overlaps, key=overlaps.get)
            best_score = overlaps[best_match]

            st.session_state.hop_draw_result = {
                "input": noisy, "output": recovered,
                "match": best_match, "score": best_score,
                "overlaps": overlaps, "energies": energies
            }

    with c2:
        section("🔍 Recognition Result", "What the network sees")
        result = st.session_state.get("hop_draw_result", None)
        if result:
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(pattern_heatmap(result["input"], "Your Drawing (10×10)", 180), use_container_width=True)
            with r2:
                st.plotly_chart(pattern_heatmap(result["output"], "Network Recall", 180), use_container_width=True)

            score_color = "#00ffcc" if result["score"] > 0.6 else "#ffb347" if result["score"] > 0.3 else "#ff5580"
            st.markdown(f"""
            <div class="card card-glow pulse" style="text-align:center;padding:1.5rem;">
              <div style="font-family:var(--mono);font-size:0.7rem;color:var(--text-dim);letter-spacing:0.2em;margin-bottom:0.5rem;">BEST MATCH</div>
              <div style="font-family:var(--display);font-size:3rem;font-weight:800;color:{score_color};">{result['match']}</div>
              <div style="font-family:var(--mono);font-size:1.1rem;color:{score_color};margin-top:0.3rem;">{result['score']*100:.1f}% similarity</div>
            </div>
            """, unsafe_allow_html=True)

            section("Pattern Overlaps", "Similarity to all stored memories")
            keys = list(result["overlaps"].keys())[:8]
            vals = [result["overlaps"][k] for k in keys]
            colors = ['#00d4ff' if v > 0.6 else '#3a6680' if v < 0.3 else '#0077aa' for v in vals]
            fig = go.Figure(go.Bar(x=keys, y=vals, marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition='outside', textfont=dict(color='#7ab8d4', size=10)))
            fig.add_hline(y=0.6, line_dash="dot", line_color="#00ffcc", annotation_text="Match Threshold", annotation_font=dict(color="#00ffcc", size=9))
            style_fig(fig, h=220, title="")
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
            fig.update_yaxes(range=[-0.2, 1.3])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="card" style="text-align:center;padding:3rem;"><div style="font-size:3rem;margin-bottom:1rem;">✏️</div><div style="color:var(--text-mid);font-size:1rem;">Draw on the whiteboard and click <b>Recognize</b></div></div>', unsafe_allow_html=True)

    st.divider()
    section("📚 Pattern Library", "Characters the network has memorised")
    keys = list(st.session_state.hop_memories.keys())
    for row_start in range(0, len(keys), 7):
        cols = st.columns(7)
        for ci, k in enumerate(keys[row_start:row_start+7]):
            if ci < len(cols):
                with cols[ci]:
                    st.plotly_chart(pattern_heatmap(st.session_state.hop_memories[k], k, 120), use_container_width=True)
    ai_tutor_widget("How does the network recognize my drawing?", "Hopfield pattern recognition from canvas input", "draw")

# ═══════════════════════════════════════════════════════════════
# PAGE 4: ENERGY & ATTRACTORS
# ═══════════════════════════════════════════════════════════════
def page_energy():
    hero("Energy & Attractors", "The Hopfield network is a physical system. Every state has an energy, and the network always rolls downhill toward stored memories.", eyebrow="Module 04 · Physics of Memory", badge="Step 4 of 5")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 4)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("The Energy Function", "Borrowed from statistical physics")
        st.markdown(r"""
        The **energy** of state $\mathbf{s}$:
        $$E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T \mathbf{W} \mathbf{s}$$

        **Key insights:**
        - **Lower energy = more stable** (a memory attractor)
        - Stored patterns are **local energy minima**
        - Each async update **decreases or maintains** energy
        - The network **always converges** — never oscillates!
        """)
        section("Attractors", "What are they?")
        st.markdown("""
        | Type | Description |
        |------|-------------|
        | **Stored memory** | Pattern you trained. Lowest energy. |
        | **Spurious state** | Mixture of patterns. Unintended but stable. |
        | **Reverse memory** | Negation of stored pattern (also stable!). |
        """)
        callout("🔍 Spurious states are 'false memories'. If you store A, B, C — the network may also create stable mixtures.", kind="warn", title="Spurious attractors")

    with c2:
        section("Energy Landscape Simulation", "")
        n_pats = st.slider("Patterns stored", 1, min(len(DEFAULT_PATTERNS), 14), 3)
        noise_lvl = st.slider("Noise level (flipped bits)", 0, 40, 15)
        nets = HopfieldNetwork(GRID_SIZE**2)
        chosen = list(DEFAULT_PATTERNS.values())[:n_pats]
        nets.train(chosen)
        baseline = chosen[0].copy()
        corrupt = baseline.copy()
        if noise_lvl > 0:
            flip_idx = np.random.choice(GRID_SIZE**2, min(noise_lvl, GRID_SIZE**2), replace=False)
            for i in flip_idx: corrupt[i] *= -1
        energies = []
        t_vals = np.linspace(0, 1, 40)
        for t in t_vals:
            interp = np.where(np.random.random(GRID_SIZE**2) < t, baseline, corrupt).astype(float)
            energies.append(nets.energy(interp))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_vals, y=energies, mode='lines', line=dict(color='#00d4ff', width=3), fill='tozeroy', fillcolor='rgba(0,212,255,0.06)', name='Energy'))
        min_idx = int(np.argmin(energies))
        fig.add_trace(go.Scatter(x=[t_vals[min_idx]], y=[energies[min_idx]], mode='markers', marker=dict(color='#00ffcc', size=12, symbol='diamond'), name='Attractor'))
        style_fig(fig, h=280, title="Energy Landscape (1D projection)")
        fig.update_xaxes(title="← Corrupted    Progress    Memory →")
        fig.update_yaxes(title="Energy E(s)")
        st.plotly_chart(fig, use_container_width=True)
        cap = int(0.138 * GRID_SIZE**2)
        metric_row([(n_pats, "Stored"), (cap, "Capacity"), (f"{n_pats/cap*100:.0f}%", "Used")])

    st.divider()
    section("Convergence Proof", "Why the network always stabilises")
    cols = st.columns(3)
    for col, title, body, icon in [
        (cols[0], "Async Updates", "One neuron at a time. Each update can only lower energy.", "📉"),
        (cols[1], "Bounded Below", "Energy has a minimum. It can't drop forever.", "🔒"),
        (cols[2], "Finite States", "With N binary neurons, only 2ᴺ states. Must reach a fixed point.", "🎯"),
    ]:
        with col:
            st.markdown(f'<div class="card card-glow" style="text-align:center;padding:1.2rem;"><span style="font-size:2rem;">{icon}</span><div style="font-family:var(--display);font-weight:600;color:#e2f4ff;margin:0.5rem 0 0.4rem;">{title}</div><div style="font-size:0.8rem;color:#7ab8d4;line-height:1.5;">{body}</div></div>', unsafe_allow_html=True)
    ai_tutor_widget("Why does energy always decrease in Hopfield networks?", "Energy function and attractors", "energy")

# ═══════════════════════════════════════════════════════════════
# PAGE 5: PLAYGROUND
# ═══════════════════════════════════════════════════════════════
def page_playground():
    hero("Neural Playground", "Store patterns, corrupt them with noise, and watch the Hopfield network reconstruct memories from chaos.", eyebrow="Module 05 · Experimentation Lab", badge="Full Playground")
    progress_steps(["Introduction", "Neurons", "Hebbian", "Draw & Recognize", "Energy", "Playground"], 5)

    with st.container():
        st.markdown('<div class="card card-glow" style="padding:1.2rem;">', unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns([1.2, 1.1, 1.1, 1])
        with h1:
            st.markdown('<div class="section-label">Pattern Library</div>', unsafe_allow_html=True)
            p_keys = list(st.session_state.hop_memories.keys())
            selected = st.selectbox("Pattern", ["— Free Draw —"] + p_keys, label_visibility="collapsed")
            if selected != "— Free Draw —":
                if st.button("📥 Load to Grid", use_container_width=True, type="primary"):
                    arr = st.session_state.hop_memories[selected]
                    st.session_state.hop_grid = pd.DataFrame((arr == 1).reshape(GRID_SIZE, GRID_SIZE))
                    st.session_state.hop_narrative = f"Pattern '{selected}' loaded."
                    st.session_state.hop_logs.append(f"[LOAD] '{selected}' loaded.")
                    st.rerun()
        with h2:
            st.markdown('<div class="section-label">Noise Injection</div>', unsafe_allow_html=True)
            noise = st.slider("Corrupt bits", 1, 50, 15, label_visibility="collapsed")
            if st.button("🔴 Inject Noise", use_container_width=True):
                flat = st.session_state.hop_grid.values.flatten().copy()
                idx = np.random.choice(GRID_SIZE**2, min(noise, GRID_SIZE**2), replace=False)
                for i in idx: flat[i] = not flat[i]
                st.session_state.hop_grid = pd.DataFrame(flat.reshape(GRID_SIZE, GRID_SIZE))
                st.session_state.hop_narrative = f"⚡ {noise} neurons flipped. Can the network recover?"
                st.session_state.hop_logs.append(f"[NOISE] {noise} bits flipped.")
                st.rerun()
        with h3:
            st.markdown('<div class="section-label">Store New Pattern</div>', unsafe_allow_html=True)
            new_name = st.text_input("Name", placeholder="e.g. My Letter", label_visibility="collapsed")
            if st.button("💾 Store", use_container_width=True, type="primary"):
                if new_name.strip():
                    flat = st.session_state.hop_grid.values.flatten()
                    st.session_state.hop_memories[new_name] = np.where(flat, 1.0, -1.0)
                    st.session_state.hop_net.train(list(st.session_state.hop_memories.values()))
                    st.session_state.hop_narrative = f"'{new_name}' stored via Hebbian learning."
                    st.session_state.hop_logs.append(f"[LEARN] '{new_name}' stored. Total: {len(st.session_state.hop_memories)}")
                    st.rerun()
        with h4:
            st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
            if st.button("🗑 Clear Grid", use_container_width=True):
                st.session_state.hop_grid = pd.DataFrame(np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool))
                st.rerun()
            if st.button("💥 Reset All", use_container_width=True):
                st.session_state.hop_memories = dict(DEFAULT_PATTERNS)
                st.session_state.hop_net.train(list(st.session_state.hop_memories.values()))
                st.session_state.hop_grid = pd.DataFrame(np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool))
                st.session_state.hop_energy_hist = []
                st.session_state.hop_logs = ["[RESET] System reset."]
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        section("Active Network Grid", "Click neurons to toggle · Blue = ON · Dark = OFF")
        changed = render_grid(st.session_state.hop_grid, key_prefix="play")
        if changed: st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            if st.button("▶ Full Recovery (60 steps)", use_container_width=True, type="primary"):
                curr = flat_state()
                st.session_state.hop_energy_hist = []
                for _ in range(60):
                    curr, _, _ = st.session_state.hop_net.update_async(curr)
                    st.session_state.hop_energy_hist.append(st.session_state.hop_net.energy(curr))
                st.session_state.hop_grid = pd.DataFrame((curr == 1).reshape(GRID_SIZE, GRID_SIZE))
                st.session_state.hop_narrative = f"Recovery complete. Energy: {st.session_state.hop_energy_hist[-1]:.3f}"
                st.session_state.hop_logs.append(f"[RUN] 60-step recovery. E={st.session_state.hop_energy_hist[-1]:.3f}")
                st.rerun()
        with r2:
            if st.button("⏭ Single Step", use_container_width=True):
                curr = flat_state()
                curr, upd, act = st.session_state.hop_net.update_async(curr)
                st.session_state.hop_grid = pd.DataFrame((curr == 1).reshape(GRID_SIZE, GRID_SIZE))
                e = st.session_state.hop_net.energy(curr)
                st.session_state.hop_energy_hist.append(e)
                st.session_state.hop_narrative = f"Neuron #{upd} updated. Activation = {act:.3f}"
                st.session_state.hop_logs.append(f"[STEP] Neuron #{upd} → {'ON' if curr[upd]==1 else 'OFF'}")
                st.rerun()

    with right:
        section("Network Status", "Real-time metrics")
        st.markdown(f'<div class="card pulse" style="min-height:60px;display:flex;align-items:center;padding:1rem 1.5rem;margin-bottom:1rem;"><span style="font-family:var(--mono);font-size:0.9rem;color:#60b8d4;line-height:1.6;">{st.session_state.hop_narrative}</span></div>', unsafe_allow_html=True)
        curr = flat_state()
        e = st.session_state.hop_net.energy(curr)
        on_n = int((curr == 1).sum())
        energy_bar(abs(e), 20, "Network Energy |E|")
        metric_row([(f"{e:.2f}", "Energy"), (on_n, "+1 Neurons"), (GRID_SIZE**2 - on_n, "−1 Neurons"), (len(st.session_state.hop_memories), "Memories")])

        section("Pattern Overlap", "Similarity to memories")
        overlaps = {k: st.session_state.hop_net.overlap(curr, v) for k, v in st.session_state.hop_memories.items()}
        if overlaps:
            keys = list(overlaps.keys())[:8]
            vals = [overlaps[k] for k in keys]
            colors = ['#00d4ff' if v > 0.6 else '#3a6680' if v < 0 else '#0077aa' for v in vals]
            fig = go.Figure(go.Bar(x=keys, y=vals, marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition='outside', textfont=dict(color='#7ab8d4', size=10)))
            fig.add_hline(y=0.9, line_dash="dot", line_color="#00ffcc", annotation_text="Recall", annotation_font=dict(color="#00ffcc", size=9))
            style_fig(fig, h=200, title="")
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
            fig.update_yaxes(range=[-1.1, 1.3])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Energy Descent", "🕸 Weight Matrix", "📋 Activity Log", "📚 Stored Patterns"])
    with tab1:
        if st.session_state.hop_energy_hist:
            fig = go.Figure(go.Scatter(y=st.session_state.hop_energy_hist, mode='lines+markers', line=dict(color='#00d4ff', width=2.5), marker=dict(size=4, color='#00ffcc'), fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'))
            style_fig(fig, h=260, title="Energy Descent During Recovery")
            fig.update_xaxes(title="Steps"); fig.update_yaxes(title="Energy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a Full Recovery to see the energy descent curve.")
    with tab2:
        fig = go.Figure(data=go.Heatmap(z=st.session_state.hop_net.W, colorscale=[[0,"#0a0e1a"],[0.5,"#001133"],[1,"#00d4ff"]], zmid=0, showscale=True, colorbar=dict(tickfont=dict(color="#7ab8d4",size=9),thickness=12)))
        style_fig(fig, h=400, title=f"W — {len(st.session_state.hop_memories)} patterns encoded")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        log_terminal(st.session_state.hop_logs)
    with tab4:
        keys = list(st.session_state.hop_memories.keys())
        for row_start in range(0, len(keys), 7):
            cols = st.columns(7)
            for ci, k in enumerate(keys[row_start:row_start+7]):
                if ci < len(cols):
                    with cols[ci]:
                        st.plotly_chart(pattern_heatmap(st.session_state.hop_memories[k], k, 120), use_container_width=True)
    ai_tutor_widget("Explain what happened during the pattern recovery", "Hopfield playground experimentation", "playground")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🧠 Hopfield<span>Lab</span></div>', unsafe_allow_html=True)
        module = st.radio("Navigation", ["🏠 Introduction", "⚡ Neurons & States", "🔗 Hebbian Learning", "🎨 Draw & Recognize", "🌋 Energy & Attractors", "🎮 Playground"], label_visibility="collapsed")
        st.session_state.hop_module = module
        st.markdown("---")
        st.markdown('<div class="section-label">Network Status</div>', unsafe_allow_html=True)
        n_mem = len(st.session_state.hop_memories)
        cap = int(0.138 * GRID_SIZE**2)
        pct = n_mem / cap * 100
        st.markdown(f'<div style="margin:0.5rem 0;font-size:0.8rem;color:#7ab8d4;">Patterns: <span style="color:#00d4ff;font-family:var(--mono);">{n_mem}</span> / {cap}</div><div class="energy-bar-wrap"><div class="energy-bar-fill" style="width:{min(pct,100):.0f}%;{"background:linear-gradient(90deg,#ff5580,#ffb347);" if pct>80 else ""}"></div></div>', unsafe_allow_html=True)
        if n_mem >= cap:
            st.warning("⚠️ Near capacity!")
        st.markdown("---")
        st.markdown(f'<div style="font-size:0.75rem;color:#3a6680;line-height:1.8;"><div style="color:#4fa8c8;margin-bottom:0.3rem;font-family:var(--mono);font-size:0.65rem;letter-spacing:0.1em;">QUICK REFERENCE</div><b style="color:#7ab8d4;">N</b> = {GRID_SIZE**2} neurons<br><b style="color:#7ab8d4;">Capacity</b> ≈ {cap}<br><b style="color:#7ab8d4;">Update</b>: sign(Ws)<br><b style="color:#7ab8d4;">Energy</b>: −½ sᵀWs<br><b style="color:#7ab8d4;">Learning</b>: W += ppᵀ/N</div>', unsafe_allow_html=True)
        ai_key = get_nvidia_key()
        st.markdown("---")
        st.markdown(f'<div style="font-size:0.7rem;color:#3a6680;text-align:center;">AI Tutor: {"🟢 Connected" if ai_key else "⚪ Add NVIDIA_API_KEY"}<br>Based on Hopfield (1982)</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    init_state()
    sidebar()

    m = st.session_state.hop_module
    if m == "🏠 Introduction": page_intro()
    elif m == "⚡ Neurons & States": page_neurons()
    elif m == "🔗 Hebbian Learning": page_hebbian()
    elif m == "🎨 Draw & Recognize": page_draw()
    elif m == "🌋 Energy & Attractors": page_energy()
    elif m == "🎮 Playground": page_playground()

if __name__ == "__main__":
    main()