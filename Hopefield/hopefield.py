"""
🧠 Hopfield Network — Ultimate Interactive Learning Module
Features: Draw & Recognize, AI Tutor with Voice, Energy Landscape, Full Playground
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time, math, os, json
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

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
# NVIDIA AI ENGINE + VOICE
# ═══════════════════════════════════════════════════════════════
def _get_key():
    k = os.getenv("NVIDIA_API_KEY")
    if not k:
        try: k = st.secrets.get("NVIDIA_API_KEY", None)
        except: pass
    return k

def ai_explain(prompt, context=""):
    key = _get_key()
    if not key: return None
    try:
        from openai import OpenAI
        c = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        r = c.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are a brilliant neural network professor. Explain Hopfield network concepts clearly with analogies and examples. Use markdown. Keep under 250 words."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ], temperature=0.7, max_tokens=500)
        return r.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI error: {e}"

def speak_text(text):
    """Inject browser TTS to read text aloud."""
    clean = text.replace("'", "\\'").replace("\n", " ").replace('"', '\\"')
    clean = clean[:1500]
    components.html(f"""
    <script>
    (function() {{
        const synth = window.parent.speechSynthesis;
        synth.cancel();
        const u = new SpeechSynthesisUtterance('{clean}');
        u.rate = 0.95; u.pitch = 1.0;
        const voices = synth.getVoices();
        const eng = voices.find(v => v.lang.startsWith('en'));
        if (eng) u.voice = eng;
        synth.speak(u);
    }})();
    </script>
    """, height=0)

def ai_tutor_block(default_q, context, key_id):
    """Full AI Tutor widget with voice output."""
    with st.expander("🤖 AI Tutor — Ask Anything", expanded=False):
        c1, c2 = st.columns([4, 1])
        with c1:
            q = st.text_input("Ask a question:", value=default_q, key=f"aiq_{key_id}", label_visibility="collapsed", placeholder=default_q)
        with c2:
            ask = st.button("🧠 Ask", key=f"aib_{key_id}", type="primary", use_container_width=True)
        if ask:
            with st.spinner("🧠 Thinking…"):
                ans = ai_explain(q, context)
            if ans:
                st.session_state[f"ai_ans_{key_id}"] = ans
            else:
                st.info("💡 Add NVIDIA_API_KEY to .env or Streamlit Secrets to enable AI Tutor.")
        ans = st.session_state.get(f"ai_ans_{key_id}", None)
        if ans:
            st.markdown(f"""<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1.2rem 1.5rem;color:#e2f4ff;font-size:0.92rem;line-height:1.75;margin-top:0.5rem;">{ans}</div>""", unsafe_allow_html=True)
            if st.button("🔊 Read Aloud", key=f"spk_{key_id}"):
                speak_text(ans)

# ═══════════════════════════════════════════════════════════════
# HOPFIELD ENGINE
# ═══════════════════════════════════════════════════════════════
class HopfieldNetwork:
    def __init__(self, size=100):
        self.N = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        self.W = np.zeros((self.N, self.N))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return -0.5 * float(s @ self.W @ s)

    def update_async(self, s, idx=None):
        s = s.copy()
        if idx is None: idx = np.random.randint(0, self.N)
        a = float(self.W[idx] @ s)
        s[idx] = 1.0 if a >= 0 else -1.0
        return s, idx, a

    def update_sync(self, s):
        return np.where(self.W @ s >= 0, 1.0, -1.0)

    def overlap(self, s, mem):
        return float(s @ mem) / self.N

    def recover(self, s, steps=80):
        h = [s.copy()]; e = [self.energy(s)]
        for _ in range(steps):
            s, _, _ = self.update_async(s)
            h.append(s.copy()); e.append(self.energy(s))
        return s, h, e

# ═══════════════════════════════════════════════════════════════
# 10×10 PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════
G = 10

def b(g):
    return np.array(g).flatten().astype(float) * 2 - 1

PATTERNS = {
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
    "✚": b([[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0]]),
    "◇": b([[0,0,0,0,1,1,0,0,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,1,0,0],[0,0,0,1,0,0,1,0,0,0],[0,0,0,0,1,1,0,0,0,0]]),
}

# ═══════════════════════════════════════════════════════════════
# THEME CSS — Transparent for live canvas
# ═══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne+Mono&family=Syne:wght@400;600;700;800&display=swap');
:root{--c0:rgba(10,22,40,0.75);--c1:rgba(13,31,56,0.7);--bd:rgba(0,212,255,0.15);--bg:rgba(0,212,255,0.4);--cy:#00d4ff;--tl:#00ffcc;--am:#ffb347;--rs:#ff5580;--t1:#e2f4ff;--t2:#7ab8d4;--t3:#3a6680;--mn:'Syne Mono',monospace;--dp:'Syne',sans-serif;--bd2:'Space Grotesk',sans-serif;}
[data-testid="stSidebar"]{background:rgba(5,15,31,0.92) !important;border-right:1px solid var(--bd);}
.hero{position:relative;padding:2.5rem 2rem;background:radial-gradient(ellipse at 20% 50%,rgba(0,212,255,0.08),transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(0,255,204,0.05),transparent 50%),var(--c0);border:1px solid var(--bd);border-radius:20px;overflow:hidden;margin-bottom:2rem;backdrop-filter:blur(12px);}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cy),var(--tl),transparent);animation:scan 3s ease-in-out infinite;}
@keyframes scan{0%,100%{opacity:.4}50%{opacity:1}}
.hero-ey{font-family:var(--mn);font-size:.72rem;letter-spacing:.25em;color:var(--cy);text-transform:uppercase;margin-bottom:.75rem;opacity:.8;}
.hero-t{font-family:var(--dp);font-size:clamp(1.8rem,4vw,3rem);font-weight:800;line-height:1.1;background:linear-gradient(135deg,#e2f4ff 30%,var(--cy) 70%,var(--tl));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem;}
.hero-s{font-size:1rem;color:var(--t2);max-width:600px;line-height:1.7;}
.crd{background:var(--c0);border:1px solid var(--bd);border-radius:14px;padding:1.5rem;margin-bottom:1rem;backdrop-filter:blur(10px);transition:border-color .3s;}
.crd:hover{border-color:var(--bg);}
.crd-glow{box-shadow:0 0 30px rgba(0,212,255,0.08),inset 0 1px 0 rgba(0,212,255,0.1);}
.slbl{font-family:var(--mn);font-size:.68rem;letter-spacing:.2em;color:var(--cy);text-transform:uppercase;margin-bottom:.3rem;}
.stit{font-family:var(--dp);font-size:1.5rem;font-weight:700;color:var(--t1);margin-bottom:.2rem;}
.sdesc{font-size:.9rem;color:var(--t2);margin-bottom:1.2rem;}
.co{padding:1.2rem 1.4rem;border-radius:10px;margin:1rem 0;border-left:3px solid;font-size:.93rem;line-height:1.65;backdrop-filter:blur(8px);}
.co-i{background:rgba(0,212,255,0.06);border-color:var(--cy);color:var(--t2);}
.co-t{background:rgba(0,255,204,0.06);border-color:var(--tl);color:var(--t2);}
.co-w{background:rgba(255,179,71,0.06);border-color:var(--am);color:#c8a070;}
.co-ttl{font-family:var(--mn);font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.4rem;opacity:.9;}
.mr{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;}
.mp{flex:1;min-width:100px;background:var(--c1);border:1px solid var(--bd);border-radius:10px;padding:1rem;text-align:center;backdrop-filter:blur(8px);}
.mp .v{font-family:var(--mn);font-size:1.6rem;font-weight:600;color:var(--cy);}
.mp .l{font-size:.72rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem;}
.stp{display:flex;gap:0;margin:1.5rem 0;}.st{flex:1;text-align:center;padding:.7rem .4rem;border-top:2px solid rgba(13,31,56,.5);font-size:.75rem;color:var(--t3);cursor:default;}
.st.ac{border-color:var(--cy);color:var(--cy);font-weight:600;}.st.dn{border-color:var(--tl);color:var(--tl);}
.stn{display:block;font-family:var(--mn);font-size:.6rem;opacity:.6;margin-bottom:.2rem;}
.ebw{background:var(--c1);border-radius:99px;height:10px;overflow:hidden;margin:.5rem 0;border:1px solid var(--bd);}
.ebf{height:100%;border-radius:99px;transition:width .5s ease;background:linear-gradient(90deg,#00a8cc,#00d4ff,#00ffcc);}
.lt{background:rgba(1,10,20,0.8);border:1px solid var(--bd);border-radius:10px;padding:1rem;font-family:var(--mn);font-size:.78rem;color:#4fa8c8;max-height:200px;overflow-y:auto;line-height:1.8;backdrop-filter:blur(8px);}
.ll{margin:0;opacity:.85;}.ll:last-child{color:var(--cy);opacity:1;}.ll::before{content:'› ';color:var(--tl);}
.cg{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0;}
.cc{background:var(--c1);border:1px solid var(--bd);border-radius:12px;padding:1.2rem;text-align:center;transition:.2s;backdrop-filter:blur(8px);}
.cc:hover{border-color:var(--bg);transform:translateY(-2px);}
.ci{font-size:2rem;margin-bottom:.6rem;display:block;}.cn{font-family:var(--dp);font-size:.92rem;font-weight:600;color:var(--t1);margin-bottom:.4rem;}.cd{font-size:.78rem;color:var(--t3);line-height:1.5;}
.sb{font-family:var(--dp);font-weight:800;font-size:1.1rem;color:var(--cy);padding:.5rem 0 1.5rem;border-bottom:1px solid var(--bd);margin-bottom:1.5rem;}.sb span{color:var(--tl);}
.badge{display:inline-block;padding:.2em .7em;border-radius:99px;font-family:var(--mn);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;background:rgba(0,212,255,0.12);color:var(--cy);border:1px solid rgba(0,212,255,0.25);}
div.stButton > button[kind="primary"]{background:linear-gradient(135deg,#0077aa,#00a8cc) !important;border:none !important;color:white !important;font-weight:600 !important;box-shadow:0 4px 20px rgba(0,168,204,0.3) !important;border-radius:8px !important;height:38px !important;width:auto !important;padding:0 1.2rem !important;font-size:.88rem !important;}
div.stButton > button[kind="primary"]:hover{box-shadow:0 4px 28px rgba(0,212,255,0.5) !important;}
@keyframes pg{0%,100%{box-shadow:0 0 8px rgba(0,212,255,0.3);}50%{box-shadow:0 0 24px rgba(0,212,255,0.7);}}.pulse{animation:pg 2s ease-in-out infinite;}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{font-family:var(--dp);color:var(--t1);}
[data-testid="stTabs"] button{font-family:var(--bd2) !important;color:var(--t3) !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--cy) !important;border-bottom-color:var(--cy) !important;}
hr{border-color:var(--bd) !important;}
</style>
"""

# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,40,0.5)", font=dict(family="Space Grotesk", color="#7ab8d4"), margin=dict(l=10,r=10,t=30,b=10))

def sfig(fig, h=250, t=""):
    fig.update_layout(**PL, height=h, title=dict(text=t, font=dict(color="#e2f4ff", size=13)))
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.1)")

def pheat(arr, t="", h=200):
    fig = go.Figure(data=go.Heatmap(z=arr.reshape(G,G), colorscale=[[0,"#0a1628"],[1,"#00d4ff"]], showscale=False, zmin=-1, zmax=1))
    sfig(fig, h=h, t=t); fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def hero(title, sub, ey="", badge=""):
    bh = f'<span class="badge" style="margin-bottom:1rem;display:inline-block;">{badge}</span><br>' if badge else ""
    st.markdown(f'<div class="hero">{bh}<div class="hero-ey">{ey}</div><div class="hero-t">{title}</div><div class="hero-s">{sub}</div></div>', unsafe_allow_html=True)

def sec(lbl, ttl, desc=""):
    d = f"<div class='sdesc'>{desc}</div>" if desc else ""
    st.markdown(f'<div class="slbl">{lbl}</div><div class="stit">{ttl}</div>{d}', unsafe_allow_html=True)

def cot(txt, k="i", t=""):
    th = f'<div class="co-ttl">{t}</div>' if t else ""
    st.markdown(f'<div class="co co-{k}">{th}{txt}</div>', unsafe_allow_html=True)

def mrow(items):
    ps = "".join(f'<div class="mp"><div class="v">{v}</div><div class="l">{l}</div></div>' for v,l in items)
    st.markdown(f'<div class="mr">{ps}</div>', unsafe_allow_html=True)

def psteps(steps, cur):
    h = "<div class='stp'>"
    for i,s in enumerate(steps):
        c = "dn" if i < cur else ("ac" if i == cur else "")
        h += f'<div class="st {c}"><span class="stn">0{i+1}</span>{s}</div>'
    st.markdown(h + "</div>", unsafe_allow_html=True)

def ebar(v, mx, lbl="Energy"):
    p = min(max(abs(v)/mx*100,0),100)
    st.markdown(f'<div style="margin:.5rem 0;"><div style="display:flex;justify-content:space-between;font-size:.78rem;color:#7ab8d4;margin-bottom:4px;"><span>{lbl}</span><span style="font-family:var(--mn);color:#00d4ff;">{v:.3f}</span></div><div class="ebw"><div class="ebf" style="width:{p}%;"></div></div></div>', unsafe_allow_html=True)

def logt(logs):
    ls = "".join(f'<p class="ll">{l}</p>' for l in logs[-20:])
    st.markdown(f'<div class="lt">{ls}</div>', unsafe_allow_html=True)

def ccgrid(items):
    h = '<div class="cg">'
    for ic,nm,ds in items:
        h += f'<div class="cc"><span class="ci">{ic}</span><div class="cn">{nm}</div><div class="cd">{ds}</div></div>'
    st.markdown(h + "</div>", unsafe_allow_html=True)

def rgrid(df, kp="n"):
    ch = False
    for r in range(G):
        cols = st.columns(G)
        for c in range(G):
            on = bool(df.iloc[r,c])
            with cols[c]:
                st.markdown(f'<div class="{"neuron-on" if on else ""}">', unsafe_allow_html=True)
                if st.button("·", key=f"{kp}_{r}_{c}"):
                    df.iloc[r,c] = not on; ch = True
                st.markdown("</div>", unsafe_allow_html=True)
    return ch

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init():
    d = {"hop_mem": dict(PATTERNS), "hop_grid": pd.DataFrame(np.zeros((G,G), dtype=bool)),
         "hop_nrr": "Awaiting input…", "hop_ehist": [], "hop_mod": "🏠 Introduction",
         "hop_log": ["[BOOT] Hopfield Engine v3 loaded.", f"[INFO] {len(PATTERNS)} patterns ready."]}
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k] = v
    if "hop_net" not in st.session_state:
        n = HopfieldNetwork(G*G); n.train(list(st.session_state.hop_mem.values()))
        st.session_state.hop_net = n

def fstate():
    return np.where(st.session_state.hop_grid.values.flatten(), 1.0, -1.0)

# ═══════════════════════════════════════════════════════════════
# PAGE 0: INTRODUCTION
# ═══════════════════════════════════════════════════════════════
def p_intro():
    hero("Hopfield Network", "An interactive journey into associative memory — how neurons store, recall, and reconstruct patterns from noise.", ey="Neural Memory Lab · Learning Module", badge="Beginner Friendly")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        sec("00 · The Big Idea", "What is a Hopfield Network?")
        st.markdown("""
        Imagine struggling to remember a friend's face. Even from a blurry photo,
        you instantly recognise them. That's **associative memory**.

        A **Hopfield Network** (1982) works this way:
        1. **Learns** patterns by adjusting connection strengths
        2. **Recalls** stored patterns from corrupted/incomplete versions
        3. **Converges** to a stable state — like a ball rolling into a valley
        """)
        cot("💡 One of the first neural networks fully explainable with physics. It bridged neuroscience, physics, and computer science.", k="t", t="Why does it matter?")
        sec("01 · Key Concepts", "Three ideas power the whole system")
        ccgrid([("⚡","Neurons","Binary units: ON (+1) or OFF (−1). Like pixels."),("🔗","Synaptic Weights","Connection strengths. Positive=agree, Negative=disagree."),("🏔️","Energy","Stability measure. Network always minimises energy.")])
    with c2:
        sec("02 · Real-World Analogies", "")
        for ic,tt,bd in [("🧲","Magnetic Memory","Neurons snap into stable configurations like magnetic domains."),("📸","Photo Restoration","Given a scratched photo, fills in missing parts from memory."),("🌊","Ball in a Valley","Energy minimisation = ball rolling to the lowest valley."),("🔒","Content-Addressable","Data→data. The pattern IS the address.")]:
            st.markdown(f'<div class="crd" style="padding:1rem 1.2rem;margin-bottom:.7rem;"><span style="font-size:1.4rem;">{ic}</span><span style="font-family:var(--dp);font-weight:600;color:#e2f4ff;margin-left:.5rem;">{tt}</span><div style="font-size:.82rem;color:#7ab8d4;margin-top:.4rem;line-height:1.5;">{bd}</div></div>', unsafe_allow_html=True)
    st.divider()
    sec("03 · Learning Path", "Six modules from zero to expert")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 0)
    cot("Use the sidebar to navigate between modules.", k="i", t="How to use")
    ai_tutor_block("What makes Hopfield networks special?", "Introduction to Hopfield networks", "intro")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: NEURONS
# ═══════════════════════════════════════════════════════════════
def p_neurons():
    hero("Neurons & Binary States", "The simplest unit: a neuron that is either ON or OFF.", ey="Module 01 · Foundations", badge="Step 1/5")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 1)
    c1, c2 = st.columns([1, 1])
    with c1:
        sec("What is a Neuron?", "The atom of the network")
        st.markdown("""
        | State | Value | Meaning |
        |-------|-------|---------|
        | **Firing** | `+1` | Active (bright pixel) |
        | **Silent** | `−1` | Inactive (dark pixel) |

        For a 10×10 grid: **100 neurons**. Combined state = a pattern.
        """)
        cot("Why +1/−1? The math is cleaner! Energy and Hebbian learning simplify beautifully.", k="t", t="Math note")
        sec("Update Rule", "")
        st.markdown(r"""
        $$h_i = \sum_{j \neq i} W_{ij} \cdot s_j \qquad s_i^{new} = \begin{cases} +1 & h_i \geq 0 \\ -1 & h_i < 0 \end{cases}$$
        Connected neurons **vote** on each neuron's state, weighted by connection strength.
        """)
    with c2:
        sec("Interactive Demo", "Click neurons to toggle ON/OFF")
        dk = "dg10"
        if dk not in st.session_state: st.session_state[dk] = pd.DataFrame(np.zeros((G,G), dtype=bool))
        dg = st.session_state[dk]
        if rgrid(dg, kp="dm"): st.session_state[dk] = dg; st.rerun()
        on = int(dg.values.sum())
        mrow([(on, "+1 Active"), (G*G-on, "−1 Silent"), (f"{on/G/G*100:.0f}%", "Density")])
    st.divider()
    with st.expander("🧪 Quiz"):
        st.markdown("**Q: How many states can 100 neurons have?**\n\n✅ 2¹⁰⁰ ≈ 1.27 × 10³⁰ — but only ~14 can be stored reliably.")
    ai_tutor_block("Explain the update rule with a simple example", "Neurons & binary states", "neur")

# ═══════════════════════════════════════════════════════════════
# PAGE 2: HEBBIAN LEARNING
# ═══════════════════════════════════════════════════════════════
def p_hebbian():
    hero("Hebbian Learning", '"Neurons that fire together, wire together." — Hebb, 1949', ey="Module 02 · Learning Rule", badge="Step 2/5")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 2)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        sec("The Learning Rule", "How patterns are etched into weights")
        st.markdown(r"""
        $$W \leftarrow W + \frac{1}{N}\mathbf{p} \cdot \mathbf{p}^T, \qquad W_{ii} = 0$$

        **Intuitively:** Both ON → strengthen. Disagreeing → weaken. No self-connections.
        """)
        cot("One-shot learning! No gradient descent. Each pattern stored with a single matrix addition.", k="t", t="No backprop needed")
        cot("⚠️ Capacity ≈ 0.138 × N. For 100 neurons → ~14 patterns max.", k="w", t="Storage limit")
    with c2:
        sec("Live Weight Matrix", "Select patterns to see W")
        pts = st.multiselect("Patterns:", list(PATTERNS.keys()), default=list(PATTERNS.keys())[:3], key="heb_sel")
        if pts:
            tn = HopfieldNetwork(G*G); tn.train([PATTERNS[k] for k in pts])
            fig = go.Figure(data=go.Heatmap(z=tn.W, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]], zmid=0, showscale=True, colorbar=dict(tickfont=dict(color="#7ab8d4",size=9),thickness=10)))
            sfig(fig, h=320, t=f"Weight Matrix — {len(pts)} patterns"); fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)
            w = tn.W
            mrow([(f"{w.max():.3f}","Max W"),(f"{w.min():.3f}","Min W"),(f"{np.abs(w).mean():.4f}","Avg |W|")])
    st.divider()
    sec("Step-by-Step", "How storing 'N' modifies the weight matrix")
    t1,t2,t3 = st.tabs(["1: Pattern","2: Outer Product","3: Final W"])
    with t1: st.plotly_chart(pheat(PATTERNS["N"],"Letter N",200), use_container_width=True)
    with t2:
        fig = go.Figure(data=go.Heatmap(z=np.outer(PATTERNS["N"],PATTERNS["N"]), zmid=0, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]], showscale=False))
        sfig(fig, h=250, t="p × pᵀ"); fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        wf = np.outer(PATTERNS["N"],PATTERNS["N"])/(G*G); np.fill_diagonal(wf,0)
        fig = go.Figure(data=go.Heatmap(z=wf, zmid=0, colorscale=[[0,"#0a1628"],[0.5,"#0d1f38"],[1,"#00d4ff"]], showscale=False))
        sfig(fig, h=250, t="W after storing 'N'"); fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    ai_tutor_block("Why is the diagonal of W always zero?", "Hebbian learning", "hebb")

# ═══════════════════════════════════════════════════════════════
# PAGE 3: DRAW & RECOGNIZE
# ═══════════════════════════════════════════════════════════════
def p_draw():
    hero("Draw & Recognize", "Draw a character on the whiteboard. The network recalls the closest stored memory.", ey="Module 03 · Pattern Recognition", badge="Interactive Canvas")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 3)
    if not CANVAS_OK:
        st.error("Install: `pip install streamlit-drawable-canvas`"); return

    c1, c2 = st.columns([1.2, 1])
    with c1:
        sec("✏️ Whiteboard", "Draw a letter or number boldly")
        cot("Use thick strokes. Drawing is downsampled to 10×10 for recognition.", k="i")
        canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=18, stroke_color="#00d4ff", background_color="#0a1628", height=300, width=300, drawing_mode="freedraw", key="hcnv")
        ca, cb, _ = st.columns(3)
        reco = ca.button("🧠 Recognize", type="primary", use_container_width=True)
        if reco and canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('L')
            sm = img.resize((G,G), Image.Resampling.LANCZOS)
            bi = np.where(np.array(sm) > 80, 1.0, -1.0).flatten()
            net = st.session_state.hop_net
            rec, _, ens = net.recover(bi, steps=80)
            ov = {k: net.overlap(rec, v) for k,v in st.session_state.hop_mem.items()}
            bm = max(ov, key=ov.get); bs = ov[bm]
            st.session_state.hop_dr = {"i":bi,"o":rec,"m":bm,"s":bs,"ov":ov,"e":ens}
    with c2:
        sec("🔍 Result", "What the network sees")
        dr = st.session_state.get("hop_dr")
        if dr:
            r1,r2 = st.columns(2)
            with r1: st.plotly_chart(pheat(dr["i"],"Your Input",180), use_container_width=True)
            with r2: st.plotly_chart(pheat(dr["o"],"Recalled",180), use_container_width=True)
            sc = "#00ffcc" if dr["s"]>.6 else "#ffb347" if dr["s"]>.3 else "#ff5580"
            st.markdown(f'<div class="crd crd-glow pulse" style="text-align:center;padding:1.5rem;"><div style="font-family:var(--mn);font-size:.7rem;color:var(--t3);letter-spacing:.2em;margin-bottom:.5rem;">BEST MATCH</div><div style="font-family:var(--dp);font-size:3rem;font-weight:800;color:{sc};">{dr["m"]}</div><div style="font-family:var(--mn);font-size:1.1rem;color:{sc};margin-top:.3rem;">{dr["s"]*100:.1f}% match</div></div>', unsafe_allow_html=True)
            # Overlap bar chart
            ks = list(dr["ov"].keys())[:8]; vs = [dr["ov"][k] for k in ks]
            cls = ['#00d4ff' if v>.6 else '#3a6680' if v<.3 else '#0077aa' for v in vs]
            fig = go.Figure(go.Bar(x=ks, y=vs, marker_color=cls, text=[f"{v:.2f}" for v in vs], textposition='outside', textfont=dict(color='#7ab8d4',size=10)))
            fig.add_hline(y=0.6, line_dash="dot", line_color="#00ffcc")
            sfig(fig, h=200); fig.update_xaxes(tickangle=-30); fig.update_yaxes(range=[-.2,1.3])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="crd" style="text-align:center;padding:3rem;"><div style="font-size:3rem;margin-bottom:1rem;">✏️</div><div style="color:var(--t2);">Draw and click <b>Recognize</b></div></div>', unsafe_allow_html=True)
    st.divider()
    sec("📚 Pattern Library", "Stored memories")
    ks = list(st.session_state.hop_mem.keys())
    for rs in range(0, len(ks), 7):
        cols = st.columns(7)
        for ci,k in enumerate(ks[rs:rs+7]):
            if ci < len(cols):
                with cols[ci]: st.plotly_chart(pheat(st.session_state.hop_mem[k], k, 120), use_container_width=True)
    ai_tutor_block("How does pattern recognition work in this network?", "Hopfield recognition", "draw")

# ═══════════════════════════════════════════════════════════════
# PAGE 4: ENERGY
# ═══════════════════════════════════════════════════════════════
def p_energy():
    hero("Energy & Attractors", "Every state has energy. The network rolls downhill toward stored memories.", ey="Module 04 · Physics", badge="Step 4/5")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 4)
    c1, c2 = st.columns([1,1])
    with c1:
        sec("Energy Function", "From statistical physics")
        st.markdown(r"""
        $$E(\mathbf{s}) = -\frac{1}{2}\mathbf{s}^T\mathbf{W}\mathbf{s}$$
        - **Lower energy = more stable** (memory attractor)
        - Each async update **decreases or maintains** energy
        - Network **always converges** — never oscillates
        """)
        sec("Attractor Types", "")
        st.markdown("""
        | Type | Description |
        |------|-------------|
        | ✅ **Stored memory** | Pattern you trained. Lowest energy. |
        | ⚠️ **Spurious** | Mixture of patterns. False memory. |
        | 🔄 **Reverse** | Negation of stored pattern (also stable). |
        """)
    with c2:
        sec("Energy Landscape", "")
        np2 = st.slider("Patterns", 1, min(len(PATTERNS),14), 3)
        nl = st.slider("Noise (flipped bits)", 0, 40, 15)
        nt = HopfieldNetwork(G*G); ch = list(PATTERNS.values())[:np2]; nt.train(ch)
        bl = ch[0].copy(); cr = bl.copy()
        if nl>0:
            fi = np.random.choice(G*G, min(nl,G*G), replace=False)
            for i in fi: cr[i] *= -1
        es = []; tv = np.linspace(0,1,40)
        for t in tv:
            ip = np.where(np.random.random(G*G)<t, bl, cr).astype(float)
            es.append(nt.energy(ip))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tv, y=es, mode='lines', line=dict(color='#00d4ff',width=3), fill='tozeroy', fillcolor='rgba(0,212,255,0.06)'))
        mi = int(np.argmin(es))
        fig.add_trace(go.Scatter(x=[tv[mi]], y=[es[mi]], mode='markers', marker=dict(color='#00ffcc',size=12,symbol='diamond'), name='Attractor'))
        sfig(fig, h=280, t="Energy Landscape")
        fig.update_xaxes(title="← Corrupted    →    Memory →"); fig.update_yaxes(title="Energy")
        st.plotly_chart(fig, use_container_width=True)
        cap = int(0.138*G*G)
        mrow([(np2,"Stored"),(cap,"Capacity"),(f"{np2/cap*100:.0f}%","Used")])
    st.divider()
    sec("Convergence Proof", "Why the network always stabilises")
    cols = st.columns(3)
    for col,tt,bd,ic in [(cols[0],"Async Updates","Each update can only lower energy.","📉"),(cols[1],"Bounded Below","Minimum exists. Can't drop forever.","🔒"),(cols[2],"Finite States","2ᴺ states. Must reach fixed point.","🎯")]:
        with col: st.markdown(f'<div class="crd crd-glow" style="text-align:center;padding:1.2rem;"><span style="font-size:2rem;">{ic}</span><div style="font-family:var(--dp);font-weight:600;color:#e2f4ff;margin:.5rem 0 .4rem;">{tt}</div><div style="font-size:.8rem;color:#7ab8d4;line-height:1.5;">{bd}</div></div>', unsafe_allow_html=True)
    ai_tutor_block("Why does energy always decrease?", "Energy and attractors", "enrg")

# ═══════════════════════════════════════════════════════════════
# PAGE 5: PLAYGROUND
# ═══════════════════════════════════════════════════════════════
def p_play():
    hero("Neural Playground", "Store, corrupt, and recover patterns. Full experimentation lab.", ey="Module 05 · Lab", badge="Full Playground")
    psteps(["Intro","Neurons","Hebbian","Draw","Energy","Playground"], 5)

    # Control bar
    st.markdown('<div class="crd crd-glow" style="padding:1.2rem;">', unsafe_allow_html=True)
    h1,h2,h3,h4 = st.columns([1.2,1.1,1.1,1])
    with h1:
        st.markdown('<div class="slbl">Pattern Library</div>', unsafe_allow_html=True)
        pk = list(st.session_state.hop_mem.keys())
        sel = st.selectbox("Pat", ["— Free Draw —"]+pk, label_visibility="collapsed")
        if sel != "— Free Draw —" and st.button("📥 Load", use_container_width=True, type="primary"):
            a = st.session_state.hop_mem[sel]
            st.session_state.hop_grid = pd.DataFrame((a==1).reshape(G,G))
            st.session_state.hop_nrr = f"'{sel}' loaded."; st.session_state.hop_log.append(f"[LOAD] '{sel}'"); st.rerun()
    with h2:
        st.markdown('<div class="slbl">Noise Injection</div>', unsafe_allow_html=True)
        ns = st.slider("Bits", 1, 50, 15, label_visibility="collapsed")
        if st.button("🔴 Inject Noise", use_container_width=True):
            fl = st.session_state.hop_grid.values.flatten().copy()
            ix = np.random.choice(G*G, min(ns,G*G), replace=False)
            for i in ix: fl[i] = not fl[i]
            st.session_state.hop_grid = pd.DataFrame(fl.reshape(G,G))
            st.session_state.hop_nrr = f"⚡ {ns} bits flipped."; st.session_state.hop_log.append(f"[NOISE] {ns}"); st.rerun()
    with h3:
        st.markdown('<div class="slbl">Store Pattern</div>', unsafe_allow_html=True)
        nn = st.text_input("Name", placeholder="My Letter", label_visibility="collapsed")
        if st.button("💾 Store", use_container_width=True, type="primary") and nn.strip():
            fl = st.session_state.hop_grid.values.flatten()
            st.session_state.hop_mem[nn] = np.where(fl, 1.0, -1.0)
            st.session_state.hop_net.train(list(st.session_state.hop_mem.values()))
            st.session_state.hop_nrr = f"'{nn}' stored."; st.session_state.hop_log.append(f"[LEARN] '{nn}'"); st.rerun()
    with h4:
        st.markdown('<div class="slbl">Actions</div>', unsafe_allow_html=True)
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.hop_grid = pd.DataFrame(np.zeros((G,G), dtype=bool)); st.rerun()
        if st.button("💥 Reset All", use_container_width=True):
            st.session_state.hop_mem = dict(PATTERNS)
            st.session_state.hop_net.train(list(PATTERNS.values()))
            st.session_state.hop_grid = pd.DataFrame(np.zeros((G,G), dtype=bool))
            st.session_state.hop_ehist = []; st.session_state.hop_log = ["[RESET]"]; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        sec("Active Grid", "Click to toggle · Blue=ON · Dark=OFF")
        if rgrid(st.session_state.hop_grid, kp="pl"): st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            if st.button("▶ Full Recovery (80 steps)", use_container_width=True, type="primary"):
                cu = fstate(); st.session_state.hop_ehist = []
                for _ in range(80):
                    cu, _, _ = st.session_state.hop_net.update_async(cu)
                    st.session_state.hop_ehist.append(st.session_state.hop_net.energy(cu))
                st.session_state.hop_grid = pd.DataFrame((cu==1).reshape(G,G))
                st.session_state.hop_nrr = f"Done. E={st.session_state.hop_ehist[-1]:.3f}"
                st.session_state.hop_log.append(f"[RUN] E={st.session_state.hop_ehist[-1]:.3f}"); st.rerun()
        with r2:
            if st.button("⏭ Single Step", use_container_width=True):
                cu = fstate(); cu, ui, ai = st.session_state.hop_net.update_async(cu)
                st.session_state.hop_grid = pd.DataFrame((cu==1).reshape(G,G))
                e = st.session_state.hop_net.energy(cu); st.session_state.hop_ehist.append(e)
                st.session_state.hop_nrr = f"Neuron #{ui} → act={ai:.3f}"
                st.session_state.hop_log.append(f"[STEP] #{ui}"); st.rerun()
    with right:
        sec("Network Status", "")
        st.markdown(f'<div class="crd pulse" style="min-height:50px;display:flex;align-items:center;padding:1rem 1.5rem;margin-bottom:1rem;"><span style="font-family:var(--mn);font-size:.9rem;color:#60b8d4;">{st.session_state.hop_nrr}</span></div>', unsafe_allow_html=True)
        cu = fstate(); e = st.session_state.hop_net.energy(cu); on = int((cu==1).sum())
        ebar(abs(e), 20, "|Energy|")
        mrow([(f"{e:.2f}","E"),(on,"+1"),(G*G-on,"−1"),(len(st.session_state.hop_mem),"Mem")])
        sec("Pattern Overlap", "")
        ov = {k: st.session_state.hop_net.overlap(cu, v) for k,v in st.session_state.hop_mem.items()}
        ks = list(ov.keys())[:8]; vs = [ov[k] for k in ks]
        cls = ['#00d4ff' if v>.6 else '#3a6680' if v<0 else '#0077aa' for v in vs]
        fig = go.Figure(go.Bar(x=ks, y=vs, marker_color=cls, text=[f"{v:.2f}" for v in vs], textposition='outside', textfont=dict(color='#7ab8d4',size=10)))
        fig.add_hline(y=0.9, line_dash="dot", line_color="#00ffcc")
        sfig(fig, h=200); fig.update_xaxes(tickangle=-30); fig.update_yaxes(range=[-1.1,1.3])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    t1,t2,t3,t4 = st.tabs(["📊 Energy","🕸 Weights","📋 Log","📚 Patterns"])
    with t1:
        if st.session_state.hop_ehist:
            fig = go.Figure(go.Scatter(y=st.session_state.hop_ehist, mode='lines+markers', line=dict(color='#00d4ff',width=2.5), marker=dict(size=4,color='#00ffcc'), fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'))
            sfig(fig, h=260, t="Energy Descent"); fig.update_xaxes(title="Steps"); fig.update_yaxes(title="E")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Run Full Recovery to see energy descent.")
    with t2:
        fig = go.Figure(data=go.Heatmap(z=st.session_state.hop_net.W, colorscale=[[0,"#0a0e1a"],[0.5,"#001133"],[1,"#00d4ff"]], zmid=0, showscale=True))
        sfig(fig, h=380, t=f"W — {len(st.session_state.hop_mem)} patterns"); fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with t3: logt(st.session_state.hop_log)
    with t4:
        ks = list(st.session_state.hop_mem.keys())
        for rs in range(0,len(ks),7):
            cols = st.columns(7)
            for ci,k in enumerate(ks[rs:rs+7]):
                if ci<len(cols):
                    with cols[ci]: st.plotly_chart(pheat(st.session_state.hop_mem[k],k,120), use_container_width=True)
    ai_tutor_block("Explain what happened during recovery", "Playground experimentation", "play")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sb">🧠 Hopfield<span>Lab</span></div>', unsafe_allow_html=True)
        m = st.radio("Nav", ["🏠 Introduction","⚡ Neurons & States","🔗 Hebbian Learning","🎨 Draw & Recognize","🌋 Energy & Attractors","🎮 Playground"], label_visibility="collapsed")
        st.session_state.hop_mod = m
        st.markdown("---")
        st.markdown('<div class="slbl">Network Status</div>', unsafe_allow_html=True)
        nm = len(st.session_state.hop_mem); cap = int(0.138*G*G); pct = nm/cap*100
        grd = "background:linear-gradient(90deg,#ff5580,#ffb347);" if pct>80 else ""
        st.markdown(f'<div style="margin:.5rem 0;font-size:.8rem;color:#7ab8d4;">Patterns: <span style="color:#00d4ff;font-family:var(--mn);">{nm}</span>/{cap}</div><div class="ebw"><div class="ebf" style="width:{min(pct,100):.0f}%;{grd}"></div></div>', unsafe_allow_html=True)
        if nm >= cap: st.warning("⚠️ Near capacity!")
        st.markdown("---")
        st.markdown(f'<div style="font-size:.75rem;color:#3a6680;line-height:1.8;"><div style="color:#4fa8c8;font-family:var(--mn);font-size:.65rem;letter-spacing:.1em;margin-bottom:.3rem;">QUICK REFERENCE</div><b style="color:#7ab8d4;">N</b>={G*G}<br><b style="color:#7ab8d4;">Capacity</b>≈{cap}<br><b style="color:#7ab8d4;">Update</b>: sign(Ws)<br><b style="color:#7ab8d4;">Energy</b>: −½sᵀWs</div>', unsafe_allow_html=True)
        ai_key = _get_key()
        st.markdown("---")
        st.markdown(f'<div style="font-size:.7rem;color:#3a6680;text-align:center;">AI: {"🟢 Active" if ai_key else "⚪ Add NVIDIA_API_KEY"}<br>🔊 Voice: Browser TTS<br>Based on Hopfield (1982)</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    st.markdown(CSS, unsafe_allow_html=True)
    init(); sidebar()
    m = st.session_state.hop_mod
    if   m == "🏠 Introduction":       p_intro()
    elif m == "⚡ Neurons & States":    p_neurons()
    elif m == "🔗 Hebbian Learning":    p_hebbian()
    elif m == "🎨 Draw & Recognize":    p_draw()
    elif m == "🌋 Energy & Attractors": p_energy()
    elif m == "🎮 Playground":          p_play()

if __name__ == "__main__":
    main()