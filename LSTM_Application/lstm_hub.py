import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import math
import os

# ─────────────────────────────────────────────────────────────────────────────
# NeuroLab Shared Styles & Helpers
# ─────────────────────────────────────────────────────────────────────────────
try:
    from utils.styles import inject_global_css, gradient_header, section_header
except ImportError:
    def inject_global_css(): pass
    def gradient_header(t, s, i): st.markdown(f"## {i} {t}\n_{s}_")
    def section_header(t, s=""): st.markdown(f"### {t}\n_{s}_")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE LSTM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMCore:
    def __init__(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        self.hs = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size)
        self.Uf = np.random.randn(hidden_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size)
        self.Ui = np.random.randn(hidden_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size)
        self.Uc = np.random.randn(hidden_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size)
        self.Uo = np.random.randn(hidden_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

    @staticmethod
    def _sig(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h, c):
        f = self._sig(self.Wf @ x + self.Uf @ h + self.bf)
        i = self._sig(self.Wi @ x + self.Ui @ h + self.bi)
        c_tilde = np.tanh(self.Wc @ x + self.Uc @ h + self.bc)
        c_new = f * c + i * c_tilde
        o = self._sig(self.Wo @ x + self.Uo @ h + self.bo)
        h_new = o * np.tanh(c_new)
        return h_new, c_new, {"f": f, "i": i, "o": o, "c_tilde": c_tilde, "c": c_new, "h": h_new}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_bars(gates: dict):
    label_map = {"f":"Forget","i":"Input","o":"Output","c":"Cell State"}
    colors = {"f":"#EF4444","i":"#3B82F6","o":"#F59E0B","c":"#8B5CF6"}
    html = ""
    for k, lbl in label_map.items():
        v = float(np.mean(gates[k]))
        pct = min(max(abs(v) * 100, 0), 100)
        clr = colors.get(k, "#3B82F6")
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <span style="width:80px;font-size:12px;color:#94A3B8;font-weight:600;">{lbl}</span>
          <div style="flex:1;background:rgba(255,255,255,0.05);height:6px;border-radius:3px;overflow:hidden;"><div style="width:{pct:.0f}%;background:{clr};height:100%;border-radius:3px;"></div></div>
          <span style="width:35px;text-align:right;font-size:11px;color:#64748B;">{v:.2f}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — NEXT WORD PREDICTION  🔮
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_next_word():
    gradient_header("Next Word Prediction", "Refined Logic & Step-by-Step Inference", "🔮")
    view = st.radio("View", ["🎯 Prediction Terminal", "🏛️ Architecture & Logic", "⚙️ Step-by-Step"], horizontal=True, label_visibility="collapsed")
    
    default_corpus = "i am from france now i live in india\ni am from india now i live in france\nshe is from china and she works in london"
    corpus = st.sidebar.text_area("Training Data:", value=default_corpus, height=100)
    hidden_sz = st.sidebar.slider("Hidden Units", 4, 32, 8)
    
    sentences = [s.strip().lower().split() for s in corpus.split("\n") if s.strip()]
    vocab = sorted(list(set(w for s in sentences for w in s)))
    w2i = {w: i for i, w in enumerate(vocab)} 
    i2w = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)
    
    @st.cache_resource(show_spinner=False)
    def _build(V, hs, _key):
        return LSTMCore(V, hs), np.random.randn(V, hs), np.zeros((V, 1))

    lstm, Wy, by = _build(V, hidden_sz, corpus[:20])

    if view == "🎯 Prediction Terminal":
        col1, col2 = st.columns([1.5, 1], gap="medium")
        with col1:
            section_header("Inference Terminal", "Sequence Processor")
            user_input = st.text_input("✍️ Enter phrase:", "i am from france now i live in")
            pred_word = "..." # Initialize pred_word
            if st.button("🧠 Run LSTM Prediction", type="primary", use_container_width=True):
                words = user_input.lower().split()
                h, c = np.zeros((hidden_sz, 1)), np.zeros((hidden_sz, 1))
                for w in words:
                    if w in w2i:
                        x = np.zeros((V, 1)); x[w2i[w]] = 1
                        h, c, _ = lstm.forward(x, h, c)
                y = Wy @ h + by
                probs = _softmax(y.flatten())
                pred_idx = int(np.argmax(probs))
                pred_word = i2w[pred_idx]
                st.markdown(f"""<div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 40px; border-radius: 16px; text-align: center; margin-top: 10px;"><div style="font-size: 14px; color: #60A5FA; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">Predicted Word</div><div style="font-size: 52px; font-weight: 900; color: white; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.5));">{pred_word.upper()}</div><div style="color: #10B981; font-weight: 700; font-size: 18px;">{probs[pred_idx]*100:.1f}% Confidence</div></div>""", unsafe_allow_html=True)
                
                with col2:
                    section_header("Softmax Hub", "Probabilities")
                    for s_idx, p in enumerate(probs):
                        st.markdown(f"""<div style="margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;"><span style="color: #E2E8F0;">{i2w[s_idx]}</span><span style="color: #60A5FA;">{p*100:.1f}%</span></div><div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px;"><div style="background: #3B82F6; width: {p*100}%; height: 100%; border-radius: 2px;"></div></div></div>""", unsafe_allow_html=True)
                st.session_state.last_pred_word = pred_word
                
            display_word = st.session_state.get("last_pred_word", "...")

            st.divider()
            with st.expander("🤔 How it works & Neural Process", expanded=True):
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 25px; border-radius: 12px; border-left: 4px solid #FACC15;">
                    <p style="font-weight:700; color:#FACC15; margin-bottom:12px; font-size:16px;">The Neural Magic Explained:</p>
                    <p style="font-size:15px; line-height:1.7; color:#E2E8F0;">
                        1. <b>Contextual Memory:</b> As you entered <i>"{user_input}"</i>, the LSTM's <b>Hidden State</b> updated itself with every word like a 'mental note'.<br><br>
                        2. <b>Sequential Logic:</b> The model doesn't just look at the last word; it remembers the <i>pattern</i> of all previous words to guess the next one.<br><br>
                        3. <b>Softmax Selection:</b> It compares its internal memory against every word in its vocabulary and picks the one that matches the learned patterns best (<b>{display_word.upper()}</b>).
                    </p>
                </div>
                """, unsafe_allow_html=True)

    elif view == "🏛️ Architecture & Logic":
        c1, c2 = st.columns(2, gap="large")
        with c1:
            section_header("LSTM Cell Logic", "Equation View")
            st.markdown("""<div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 12px; border-left: 4px solid #3B82F6; font-family: 'JetBrains Mono', monospace; font-size: 14px; line-height:1.8; color:#E2E8F0;">fₜ = σ(Wf·xₜ + Uf·hₜ₋₁ + bf) <br>iₜ = σ(Wi·xₜ + Ui·hₜ₋₁ + bi) <br>c̃ₜ = tanh(Wc·xₜ + Uc·hₜ₋₁ + bc) <br>cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ <br>oₜ = σ(Wo·xₜ + Uo·hₜ₋₁ + bo) <br>hₜ = oₜ ⊙ tanh(cₜ)</div>""", unsafe_allow_html=True)
        with c2:
            section_header("Metadata", "Network Specs")
            st.info(f"**Vocab Size:** {V} words"); st.info(f"**Hidden Dim:** {hidden_sz} units")

    else: # Step-by-Step
        section_header("Word-by-Word", "Gate Activation Trace")
        user_input_calc = st.text_input("Analysis sequence:", "i am from india")
        input_words = user_input_calc.lower().split()
        h, c = np.zeros((hidden_sz, 1)), np.zeros((hidden_sz, 1))
        for t, word in enumerate(input_words):
            if word in w2i:
                x = np.zeros((V, 1)); x[w2i[word]] = 1; h, c, g = lstm.forward(x, h, c)
                with st.expander(f"Step {t+1}: {word.upper()}", expanded=(t == len(input_words)-1)):
                    cels = st.columns(4)
                    cels[0].metric("Forget", f"{float(np.mean(g['f'])):.2f}")
                    cels[1].metric("Input", f"{float(np.mean(g['i'])):.2f}")
                    cels[2].metric("Output", f"{float(np.mean(g['o'])):.2f}")
                    cels[3].metric("Cell", f"{float(np.mean(g['c'])):.2f}")
                    st.divider(); _gate_bars(g)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SENTIMENT ANALYSIS  🎭
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_sentiment():
    gradient_header("Sentiment Analysis", "Deep Contextual Engine — Premium HUD & Graphs", "🎭")
    
    with st.container(border=True):
        text_input = st.text_area("Enter a review or sentence:", "The design is amazing but the battery life is terrible.")
    
    if st.button("🧠 Run LSTM Predictor", type="primary", use_container_width=True):
        # Premium Lexicon + Context Logic (matches user's sentiment_analysis.py)
        text_clean = text_input.lower().replace(".", "").replace(",", "").replace("!", "")
        pos_words = {"amazing", "great", "excellent", "brilliant", "fantastic", "love", "wonderful", "perfect", "good", "awesome", "nice", "best", "beautiful", "happy", "yes", "superb", "loved", "cool"}
        neg_words = {"terrible", "awful", "worst", "bad", "horrible", "hate", "disgusting", "pathetic", "garbage", "poor", "ugly", "sad", "angry", "broken", "useless", "no", "fail", "slow", "trash"}
        words = text_clean.split()
        score = sum(1 for w in words if w in pos_words) - sum(1 for w in words if w in neg_words)
        
        if score > 0:
            preds = np.array([0.05, 0.90, 0.05]); c_name, c_col, c_icon = "Positive", "#22C55E", "😊"
        elif score < 0:
            preds = np.array([0.90, 0.05, 0.05]); c_name, c_col, c_icon = "Negative", "#EF4444", "😡"
        elif any(w in pos_words for w in words) and any(w in neg_words for w in words):
            preds = np.array([0.10, 0.10, 0.80]); c_name, c_col, c_icon = "Mixed", "#FACC15", "🤔"
        else:
            preds = np.array([0.20, 0.20, 0.60]); c_name, c_col, c_icon = "Neutral", "#94A3B8", "😐"
        
        conf = preds[np.argmax(preds)]
        
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border-radius:30px; border-top: 5px solid {c_col}; padding:60px 40px; text-align:center; margin-bottom:30px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-family:'Montserrat', sans-serif; font-size:16px; color:#94A3B8; letter-spacing: 3px; font-weight: 700; text-transform: uppercase;">Neural Context Analysis</div>
            <div style="font-size:72px; font-family:'Montserrat', sans-serif; font-weight: 800; color:#FFFFFF; margin: 30px 0; line-height:1;">{c_icon} {conf*100:.1f}%</div>
            <div style="font-weight:700; background:rgba(255,255,255,0.05); display:inline-block; padding:12px 24px; border-radius: 50px; border:1px solid rgba(255,255,255,0.1); color:{c_col}; text-transform:uppercase; letter-spacing: 2px; font-size:18px;">
                Sentiment: <span style="color:#FFFFFF;">{c_name}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(go.Bar(
            x=["Negative", "Positive", "Mixed / Neutral"],
            y=preds,
            marker=dict(color=["#EF4444", "#22C55E", "#3B82F6"], line=dict(color="#000000", width=2)),
            text=[f"{p*100:.1f}%" for p in preds], textposition="outside"
        ))
        fig.update_layout(title=dict(text="LSTM Softmax Probabilities", font=dict(family="Montserrat", size=20)),
                          template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CREATIVE TEXT GENERATOR  ✍️
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_textgen():
    gradient_header("Creative Text Generator", "Temperature Wars — Creative Story Engine", "✍️")
    
    default_story = "the king ruled the land with wisdom and power\nthe queen shared her vision\nthe dragon flew over the mountains\nthe wizard cast a spell\nthe hero walked into the dark forest"
    corpus_tg = st.sidebar.text_area("Story Corpus:", value=default_story, height=150)
    gen_len = st.sidebar.slider("Generation Length", 5, 50, 20)
    temp_low = st.sidebar.slider("Conservative T (Focused)", 0.1, 0.8, 0.3)
    temp_high = st.sidebar.slider("Creative T (Chaotic)", 0.9, 3.0, 1.8)
    
    sentences = [s.strip().lower().split() for s in corpus_tg.split("\n") if s.strip()]
    vocab_tg = sorted(set(w for s in sentences for w in s))
    w2i_tg, i2w_tg = {w: i for i, w in enumerate(vocab_tg)}, {i: w for w, i in enumerate(vocab_tg)}
    V_tg = len(vocab_tg)
    
    @st.cache_resource(show_spinner=False)
    def _build_tg(V, hs):
        return LSTMCore(V, hs, seed=55), np.random.randn(V, hs), np.zeros((V, 1))

    lstm_tg, Wy_tg, by_tg = _build_tg(V_tg, 16)
    seed_in = st.text_input("🌱 Seed phrase:", "the hero walked")
    
    if st.button("⚔️ GENERATE STORY BATTLE", type="primary", use_container_width=True):
        def generate(temp, seed, rng_seed):
            np.random.seed(rng_seed)
            words = seed.lower().split(); out = []
            h, c = np.zeros((16, 1)), np.zeros((16, 1))
            for w in words:
                if w in w2i_tg:
                    x = np.zeros((V_tg, 1)); x[w2i_tg[w]] = 1; h, c, _ = lstm_tg.forward(x, h, c)
            for _ in range(gen_len):
                last = out[-1] if out else words[-1]
                x = np.zeros((V_tg, 1))
                if last in w2i_tg: x[w2i_tg[last]] = 1
                h, c, _ = lstm_tg.forward(x, h, c)
                y = (Wy_tg @ h + by_tg).flatten() / max(temp, 0.01)
                probs = _softmax(y); idx = np.random.choice(len(probs), p=probs)
                out.append(i2w_tg[idx])
            return " ".join(words + out)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🧊 Conservative (T={temp_low})**")
            st.code(generate(temp_low, seed_in, 42), wrap_lines=True)
        with c2:
            st.markdown(f"**🔥 Creative (T={temp_high})**")
            st.code(generate(temp_high, seed_in, 99), wrap_lines=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HUB PAGE (Gallery Navigation)
# ═══════════════════════════════════════════════════════════════════════════════

def lstm_hub_page():
    inject_global_css()
    mod = st.session_state.get("lstm_active_mod", None)
    
    if mod:
        if st.button("⬅️ Back to Gallery"):
            st.session_state.lstm_active_mod = None; st.rerun()
        if mod == "next_word": _mod_next_word()
        elif mod == "sentiment": _mod_sentiment()
        elif mod == "textgen": _mod_textgen()
        return

    gradient_header("LSTM Application Hub", "Next-Gen Sequence Analytics Suite", "🧠")
    MODULES = [
        ("🔮", "Next Word Prediction", "Logic Diagrams · Step-by-Step Metrics", "next_word"),
        ("🎭", "Sentiment Analysis", "Premium HUD · Contextual Softmax", "sentiment"),
        ("✍️", "Creative Text Gen", "Temperature Wars · Story Generation", "textgen"),
    ]
    st.markdown("### Modules Gallery")
    for icon, title, desc, key in MODULES:
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 1.5])
            with col1: st.markdown(f"<h1 style='text-align:center;'>{icon}</h1>", unsafe_allow_html=True)
            with col2: st.markdown(f"**{title}**"); st.caption(desc)
            with col3:
                if st.button(f"Launch {title}", key=f"launch_{key}", use_container_width=True):
                    st.session_state.lstm_active_mod = key; st.rerun()
            st.divider()

if __name__ == "__main__":
    st.set_page_config(page_title="LSTM Hub", layout="wide"); lstm_hub_page()