import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
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

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — NEXT WORD PREDICTION  🔮
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_next_word():
    inject_global_css()
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
    def _build(V, hs, _key_hash):
        return LSTMCore(V, hs), (np.random.randn(V, hs) * 0.1), np.zeros((V, 1))

    corpus_hash = hashlib.md5(corpus.encode()).hexdigest()
    lstm, Wy, by = _build(V, hidden_sz, corpus_hash)

    if view == "🎯 Prediction Terminal":
        col1, col2 = st.columns([1.5, 1], gap="medium")
        with col1:
            section_header("Inference Terminal", "Sequence Processor")
            user_input = st.text_input("✍️ Enter phrase:", "i am from france now i live in", key="next_word_input")
            pred_word_init = "..."
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
                st.session_state.last_pred_word = pred_word
                
            display_word = st.session_state.get("last_pred_word", pred_word_init)
            
            with col2:
                section_header("Softmax Hub", "Probabilities")
                if "last_pred_word" in st.session_state:
                    words = user_input.lower().split()
                    h, c = np.zeros((hidden_sz, 1)), np.zeros((hidden_sz, 1))
                    for w in words:
                        if w in w2i:
                            x = np.zeros((V, 1)); x[w2i[w]] = 1
                            h, c, _ = lstm.forward(x, h, c)
                    y = Wy @ h + by
                    current_probs = _softmax(y.flatten())
                    for s_idx, p in enumerate(current_probs):
                        st.markdown(f"""<div style="margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;"><span style="color: #E2E8F0;">{i2w[s_idx]}</span><span style="color: #60A5FA;">{p*100:.1f}%</span></div><div style="background: rgba(255,255,255,0.05); height: 4px; border-radius: 2px;"><div style="background: #3B82F6; width: {p*100}%; height: 100%; border-radius: 2px;"></div></div></div>""", unsafe_allow_html=True)

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
        user_input_calc = st.text_input("Analysis sequence:", "i am from india", key="step_input")
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

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SENTIMENT ANALYSIS  🎭
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_sentiment():
    inject_global_css()
    gradient_header("Sentiment Analysis", "Lexicon + Softmax Confidence Engine", "🎭")
    
    st.sidebar.markdown("### 📊 Lexicon Weights")
    lexicon = {
        "amazing": 1.2, "excellent": 1.0, "happy": 0.8, "good": 0.5, "love": 1.1, "best": 0.9,
        "bad": -0.8, "worst": -1.2, "terrible": -1.1, "sad": -0.7, "hate": -1.0, "awful": -0.9,
        "ok": 0.1, "average": 0.0, "fine": 0.2, "normal": 0.1, "neutral": 0.0
    }
    
    txt = st.text_area("✍️ Analysis Input:", "The NeuroLab LSTM project is absolutely amazing and powerful!")
    
    if txt:
        score = 0
        words = txt.lower().split()
        for w in words:
            score += lexicon.get(w, 0)
        
        preds = _softmax(np.array([max(0, score), max(0, -score), 0.5]))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div style="background:rgba(34,197,94,0.1); border:1px solid #22C55E; padding:20px; border-radius:12px; text-align:center;"><div style="font-size:32px;">😊</div><div style="color:#22C55E; font-weight:700;">POSITIVE</div><div style="font-size:24px; color:white;">{preds[0]*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div style="background:rgba(239,68,68,0.1); border:1px solid #EF4444; padding:20px; border-radius:12px; text-align:center;"><div style="font-size:32px;">😡</div><div style="color:#EF4444; font-weight:700;">NEGATIVE</div><div style="font-size:24px; color:white;">{preds[1]*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div style="background:rgba(59,130,246,0.1); border:1px solid #3B82F6; padding:20px; border-radius:12px; text-align:center;"><div style="font-size:32px;">🤔</div><div style="color:#3B82F6; font-weight:700;">MIXED/NEUTRAL</div><div style="font-size:24px; color:white;">{preds[2]*100:.1f}%</div></div>""", unsafe_allow_html=True)
            
        fig = go.Figure(go.Bar(
            x=["Positive", "Negative", "Neutral"], y=preds,
            marker=dict(color=["#22C55E", "#EF4444", "#3B82F6"]),
            text=[f"{p*100:.1f}%" for p in preds], textposition="outside"
        ))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CREATIVE TEXT GENERATOR  ✍️
# ═══════════════════════════════════════════════════════════════════════════════

def _mod_textgen():
    inject_global_css()
    gradient_header("Creative Text Generator", "Temperature Wars — Conservative vs Creative Battle", "✍️")
 
    st.sidebar.markdown("### 📜 Story Corpus")
    default_story = (
        "the king ruled the land with wisdom and power\n"
        "the queen stood beside the king and shared her vision\n"
        "the dragon flew over the mountains at night\n"
        "the wizard cast a spell to protect the village\n"
        "the hero walked into the dark forest alone\n"
        "the princess found the hidden door in the castle\n"
        "the knight defeated the enemy with one blow\n"
        "the ancient map led to the buried treasure\n"
        "the ship sailed through the storm without fear\n"
        "magic filled the air when the spell was spoken"
    )
    corpus_tg = st.sidebar.text_area("Corpus (one sentence per line):", value=default_story, height=200)
    hidden_tg = st.sidebar.slider("Hidden Units", 6, 32, 12, key="tg_hs")
    gen_len = st.sidebar.slider("Words to Generate", 5, 30, 15, key="tg_gl")
    temp_low = st.sidebar.slider("Conservative Temperature", 0.1, 0.8, 0.3, 0.05, key="tg_tl")
    temp_high = st.sidebar.slider("Creative Temperature", 0.9, 3.0, 1.8, 0.1, key="tg_th")
    show_word_probs = st.sidebar.checkbox("📊 Show Per-Step Probabilities", value=True)
 
    sentences = [s.strip().lower().split() for s in corpus_tg.split("\n") if s.strip()]
    if not sentences:
        st.error("Add at least one sentence.")
        return
 
    vocab_tg = sorted(set(w for s in sentences for w in s))
    w2i_tg = {word: i for i, word in enumerate(vocab_tg)}
    i2w_tg = {i: word for i, word in enumerate(vocab_tg)}
    V_tg = len(vocab_tg)
 
    @st.cache_resource(show_spinner=False)
    def _build_tg(V, hs, _key_hash):
        np.random.seed(55)
        lstm = LSTMCore(V, hs, seed=55)
        Wy = (np.random.randn(V, hs) * 0.1)
        by = np.zeros((V, 1))
        return lstm, Wy, by
 
    corpus_hash_tg = hashlib.md5(corpus_tg.encode()).hexdigest()
    lstm_tg, Wy_tg, by_tg = _build_tg(V_tg, hidden_tg, corpus_hash_tg)
 
    def one_hot_tg(w):
        v = np.zeros((V_tg, 1))
        if w in w2i_tg:
            v[w2i_tg[w]] = 1
        return v
 
    def generate(seed_text, length, temperature, rng_seed=0):
        words = seed_text.lower().split()
        if not words: return [], []
        h, c = np.zeros((hidden_tg, 1)), np.zeros((hidden_tg, 1))
        for w in words:
            if w in w2i_tg:
                h, c, _ = lstm_tg.forward(one_hot_tg(w), h, c)
 
        rng = np.random.RandomState(rng_seed)
        generated, step_probs = list(words), []
        for _ in range(length):
            last = generated[-1] if generated else words[-1]
            x = one_hot_tg(last)
            h, c, _ = lstm_tg.forward(x, h, c)
            y = (Wy_tg @ h + by_tg).flatten() / max(temperature, 0.01)
            probs = _softmax(y)
            probs = probs / (np.sum(probs) + 1e-10) 
            idx = int(rng.choice(len(probs), p=probs))
            generated.append(i2w_tg[idx])
            step_probs.append({"word": i2w_tg[idx], "confidence": float(probs[idx]), "top": i2w_tg[int(np.argmax(probs))]})
        return generated[len(words):], step_probs
 
    # ── UI ───────────────────────────────────────────────────────────────────
    seed_in = st.text_input("🌱 Seed phrase:", "the hero walked into", key="tg_seed_phrase")
    col_btn1, col_btn2 = st.columns(2)
    run_conservative = col_btn1.button(f"🧊 Conservative (T={temp_low})", use_container_width=True)
    run_creative = col_btn2.button(f"🔥 Creative (T={temp_high})", use_container_width=True)
    run_battle = st.button("⚔️ TEMPERATURE WARS — Generate Both!", type="primary", use_container_width=True)
 
    do_run = run_battle or run_conservative or run_creative
 
    if do_run:
        show_con = run_battle or run_conservative
        show_cre = run_battle or run_creative
        col_con, col_cre = st.columns(2) if run_battle else (st.container(), st.container())
 
        results = []
        if show_con:
            gen_c, sp_c = generate(seed_in, gen_len, temp_low, rng_seed=42)
            results.append((gen_c, sp_c, temp_low, "🧊 Conservative", "#3B82F6", col_con if run_battle else st))
        if show_cre:
            gen_h, sp_h = generate(seed_in, gen_len, temp_high, rng_seed=99)
            results.append((gen_h, sp_h, temp_high, "🔥 Creative", "#EC4899", col_cre if run_battle else st))
 
        for gen_words, step_probs, temp, label, clr, col in results:
            with col:
                seed_html = " ".join(f'<span style="color:#64748B;">{w}</span>' for w in seed_in.split())
                gen_html = " ".join(f'<span style="color:{clr};opacity:{0.5 + sp["confidence"]*0.5:.2f};">{w}</span>' for w, sp in zip(gen_words, step_probs))
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid {clr}44;border-radius:14px;padding:20px;margin-bottom:12px;">
                  <div style="font-size:12px;color:{clr};font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">{label}</div>
                  <div style="font-size:1.05em;line-height:1.8;font-family:'JetBrains Mono',monospace;">{seed_html} {gen_html}</div>
                  <div style="margin-top:12px;font-size:11px;color:#64748B;">Temperature: {temp} · {len(gen_words)} words generated</div>
                </div>""", unsafe_allow_html=True)
 
                if show_word_probs and step_probs:
                    confs = [sp["confidence"] for sp in step_probs]
                    fig_sp = go.Figure(go.Bar(x=[sp["word"] for sp in step_probs], y=confs, marker_color=[clr] * len(confs),
                        opacity=0.75, text=[f"{c:.2f}" for c in confs], textposition="outside", textfont=dict(color="#64748B", size=10)))
                    fig_sp.update_layout(template="plotly_dark", height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
                        margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(color="#475569", tickangle=-30, tickfont=dict(size=10)),
                        yaxis=dict(visible=False), title=dict(text="Step Confidence", font=dict(size=11, color="#64748B"), x=0))
                    col.plotly_chart(fig_sp, use_container_width=True)
 
    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.1); padding: 25px; border-radius: 12px; border-left: 4px solid #3B82F6; margin-top:20px;">
        💡 <b>Temperature Wars</b> makes the creativity/predictability tradeoff tangible in a single side-by-side view. Low temp (0.3) picks the "mathematically safest" words, while high temp (1.8) exploration leads to wild and dreamy narratives.
    </div>""", unsafe_allow_html=True)

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