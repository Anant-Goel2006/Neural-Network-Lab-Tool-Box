import streamlit as st
import numpy as np
from utils.styles import inject_global_css, gradient_header, section_header, render_nlp_insight

def lstm_prediction_page():
    inject_global_css()
    
    gradient_header("LSTM Prediction", "Step-by-Step Language Processor", "🧠")

    # -----------------------------
    # Sidebar: Data Management
    # -----------------------------
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
        <h3 style="margin:0; font-size:16px; color:#F8FAFC; font-family:'Montserrat', sans-serif;">📂 Data & Training</h3>
    </div>
    """, unsafe_allow_html=True)    # SIDEBAR: DATA (KEEP FOR VOCAB)
    
    default_text = "i am from france now i live in india\ni am from india"
    training_data = st.sidebar.text_area("Training Sentences (one per line):", value=default_text, height=100)
    
    sentences = [s.strip().lower().split() for s in training_data.split('\n') if s.strip()]
    if not sentences:
        st.error("Please provide at least one training sentence.")
        return

    vocab = sorted(list(set(word for sent in sentences for word in sent)))
    word_to_idx = {w:i for i,w in enumerate(vocab)}
    idx_to_word = {i:w for w,i in word_to_idx.items()}
    vocab_size = len(vocab)

    # -----------------------------
    # LSTM Parameters (Reset when vocab changes)
    # -----------------------------
    input_size = vocab_size
    hidden_size = 8
    np.random.seed(42) # Consistent weights for teaching

    # Initialize weights
    params = {
        'Wf': np.random.randn(hidden_size, input_size),
        'Uf': np.random.randn(hidden_size, hidden_size),
        'bf': np.zeros((hidden_size,1)),
        'Wi': np.random.randn(hidden_size, input_size),
        'Ui': np.random.randn(hidden_size, hidden_size),
        'bi': np.zeros((hidden_size,1)),
        'Wc': np.random.randn(hidden_size, input_size),
        'Uc': np.random.randn(hidden_size, hidden_size),
        'bc': np.zeros((hidden_size,1)),
        'Wo': np.random.randn(hidden_size, input_size),
        'Uo': np.random.randn(hidden_size, hidden_size),
        'bo': np.zeros((hidden_size,1)),
        'Wy': np.random.randn(vocab_size, hidden_size),
        'by': np.zeros((vocab_size,1))
    }

    # -----------------------------
    # View Selector (Dropdown)
    # -----------------------------
    view = st.selectbox("Switch View:", 
                       ["🎯 Prediction Terminal", "🏛️ Architecture & Logic", "⚙️ Step-by-Step Calculations"],
                       index=0)

    st.markdown("---")

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def tanh(x): return np.tanh(x)
    def one_hot(word):
        vec = np.zeros((vocab_size,1))
        if word in word_to_idx: vec[word_to_idx[word]] = 1
        return vec

    if view == "🏛️ Architecture & Logic":
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            section_header("The LSTM Architecture", "Four gates, one memory cell")
            st.markdown("""
            <div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; border-left: 3px solid #3B82F6;">
                <p style="margin-bottom:15px; font-weight:600; color:#3B82F6;">Mathematical Foundation</p>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px;">
                    fₜ = σ(Wf·xₜ + Uf·hₜ₋₁ + bf) <br>
                    iₜ = σ(Wi·xₜ + Ui·hₜ₋₁ + bi) <br>
                    c̃ₜ = tanh(Wc·xₜ + Uc·hₜ₋₁ + bc) <br>
                    cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ <br>
                    oₜ = σ(Wo·xₜ + Uo·hₜ₋₁ + bo) <br>
                    hₜ = oₜ ⊙ tanh(cₜ)
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            section_header("Module Capabilities", "Active Learning Profile")
            st.info(f"**Vocabulary Size:** {vocab_size} unique tokens")
            st.info(f"**Hidden Dimension:** {hidden_size} units")
            st.markdown("### Known Tokens:")
            st.write(", ".join(f"`{w}`" for w in vocab))

    elif view == "⚙️ Step-by-Step Calculations":
        user_input_calc = st.text_input("Enter sequence for step analysis:", "i am from france now i live in")
        input_words_calc = user_input_calc.lower().split()
        
        section_header("Word-by-Word Analysis", "Tracing gate activations through the sequence")
        
        h = np.zeros((hidden_size,1))
        c = np.zeros((hidden_size,1))

        if not input_words_calc:
            st.warning("Please enter some text to begin processing.")
        
        for t, word in enumerate(input_words_calc):
            if word not in word_to_idx:
                st.error(f"⚠️ '{word}' is OOV (Out of Vocabulary)")
                continue

            x = one_hot(word)
            f = sigmoid(params['Wf'] @ x + params['Uf'] @ h + params['bf'])
            i = sigmoid(params['Wi'] @ x + params['Ui'] @ h + params['bi'])
            c_tilde = tanh(params['Wc'] @ x + params['Uc'] @ h + params['bc'])
            c = f * c + i * c_tilde
            o = sigmoid(params['Wo'] @ x + params['Uo'] @ h + params['bo'])
            h = o * tanh(c)

            with st.expander(f"Time Step {t+1}: {word.upper()}", expanded=(t==len(input_words_calc)-1)):
                cols = st.columns(4)
                cols[0].metric("Forget Gate", f"{np.mean(f):.2f}")
                cols[1].metric("Input Gate", f"{np.mean(i):.2f}")
                cols[2].metric("Output Gate", f"{np.mean(o):.2f}")
                cols[3].metric("Cell Mean", f"{np.mean(c):.2f}")
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                    Hidden State (hₜ) Sample: {h.flatten()[:4].tolist()}...
                </div>
                """, unsafe_allow_html=True)

    elif view == "🎯 Prediction Terminal":
        section_header("Deep Contextual Inference", "Input a sequence to predict the next token")
        
        with st.container(border=True):
            user_input_pred = st.text_input("Enter a sequence of words:", "i am from france now i live in")
        
        if st.button("🧠 Run LSTM Prediction", type="primary", use_container_width=True):
            input_words_pred = user_input_pred.lower().split()
            
            col_main, col_probs = st.columns([1.5, 1])
            
            with col_main:
                h = np.zeros((hidden_size,1))
                c = np.zeros((hidden_size,1))

                for word in input_words_pred:
                    if word in word_to_idx:
                        x = one_hot(word)
                        f = sigmoid(params['Wf'] @ x + params['Uf'] @ h + params['bf'])
                        i = sigmoid(params['Wi'] @ x + params['Ui'] @ h + params['bi'])
                        c_tilde = tanh(params['Wc'] @ x + params['Uc'] @ h + params['bc'])
                        c = f * c + i * c_tilde
                        o = sigmoid(params['Wo'] @ x + params['Uo'] @ h + params['bo'])
                        h = o * tanh(c)

                y = params['Wy'] @ h + params['by']
                probs = np.exp(y) / np.sum(np.exp(y))
                pred_idx = np.argmax(probs)
                pred_word = idx_to_word[pred_idx]

                st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 40px; border-radius: 16px; text-align: center; margin-top: 10px; margin-bottom: 25px;">
                    <div style="font-size: 14px; color: #60A5FA; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Input Sequence Analysis Complete</div>
                    <div style="font-size: 18px; color: #94A3B8; font-style: italic; margin-bottom: 25px;">"...{' '.join(input_words_pred)}"</div>
                    <div style="font-size: 12px; color: #64748B; margin-bottom: 5px;">PROBABILITY WINNER</div>
                    <div style="font-size: 52px; font-weight: 900; color: white; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.5));">{pred_word.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🤔 What does 'Next Word' mean?", expanded=True):
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 8px; border-left: 3px solid #FACC15;">
                        <p style="font-weight:700; color:#FACC15; margin-bottom:10px;">The Neural Prediction Process:</p>
                        <p style="font-size:14px; line-height:1.6;">
                            1. <b>Context Accumulation:</b> As you entered <i>"{' '.join(input_words_pred)}"</i>, the LSTM's <b>Hidden State (h)</b> acted like a "working memory," updating itself with every new word.<br><br>
                            2. <b>The Logic Check:</b> By the time it reached the last word, the hidden state contained a mathematical representation of the entire phrase's context.<br><br>
                            3. <b>The Final Guess:</b> The model then takes this memory and maps it to the <b>Vocabulary</b> you provided in the sidebar. It calculates a probability for every known word and picks the one with the highest score (<b>{pred_word.upper()}</b>).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_probs:
                section_header("Softmax Hub", "Full Probabilities")
                for idx, p in enumerate(probs):
                    w = idx_to_word[idx]
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                            <span style="color: #E2E8F0;">{w}</span>
                            <span style="color: #60A5FA;">{p[0]*100:.1f}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px;">
                            <div style="background: #3B82F6; width: {p[0]*100}%; height: 100%; border-radius: 3px; box-shadow: 0 0 10px #3B82F6;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    lstm_prediction_page()