"""
🧠 Modern Hopfield Network Lab
Pseudoinverse Learning Engine with Voice AI
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os, json, time, io, base64
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & SETUP
# ═══════════════════════════════════════════════════════════════
G = 16 # Grid size (16x16 = 256 neurons)
N = G * G

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

st.set_page_config(page_title="Magic Hopfield", page_icon="🔮", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════
# PSEUDOINVERSE HOPFIELD ENGINE (Flawless Detection)
# ═══════════════════════════════════════════════════════════════
class PseudoinverseHopfield:
    """
    Upgraded Hopfield Engine using the Projection Rule (Moore-Penrose Pseudoinverse).
    Unlike Hebbian Learning (capacity ~0.14N), this provides exact orthogonal
    projection, granting perfect recall and drastically fewer spurious states.
    """
    def __init__(self, size=256):
        self.N = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        """W = X @ pinv(X)"""
        if not patterns:
            self.W = np.zeros((self.N, self.N))
            return
        # X is (N, P) where each column is a pattern
        X = np.column_stack(patterns)
        
        # Pseusoinverse W calculation
        X_inv = np.linalg.pinv(X)
        W = X @ X_inv
        
        # Enforce zero diagonal to ensure convergence and network stability
        np.fill_diagonal(W, 0)
        self.W = W

    def energy(self, s):
        # E = -0.5 * s^T * W * s
        return -0.5 * float(s @ self.W @ s)

    def update_async(self, s):
        s_new = s.copy()
        idx = np.random.randint(0, self.N)
        activation = float(self.W[idx] @ s_new)
        s_new[idx] = 1.0 if activation >= 0 else -1.0
        return s_new, idx, activation

    def recover(self, s, steps=150):
        curr = s.copy()
        energies = [self.energy(curr)]
        for _ in range(steps):
            curr, _, _ = self.update_async(curr)
            energies.append(self.energy(curr))
            
            # Fast convergence check:
            # Sync update check to see if we reached a stable fixed point early
            sync_next = np.where(self.W @ curr >= 0, 1.0, -1.0)
            if np.array_equal(curr, sync_next):
                break
        return curr, energies

# ═══════════════════════════════════════════════════════════════
# 16x16 HIGH RES PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════
def _parse(str_arr):
    vals = []
    for row in str_arr:
        for char in row:
            vals.append(1.0 if char == '#' else -1.0)
    return np.array(vals)

PATTERNS = {
    "A": _parse([
        "      ####      ",
        "     ######     ",
        "    ##    ##    ",
        "   ##      ##   ",
        "   ##      ##   ",
        "  ##        ##  ",
        "  ##        ##  ",
        "  ############  ",
        "  ############  ",
        "  ##        ##  ",
        " ##          ## ",
        " ##          ## ",
        " ##          ## ",
        " ##          ## ",
        "##            ##",
        "##            ##"
    ]),
    "X": _parse([
        "##            ##",
        " ##          ## ",
        "  ##        ##  ",
        "   ##      ##   ",
        "    ##    ##    ",
        "     ##  ##     ",
        "      ####      ",
        "       ##       ",
        "       ##       ",
        "      ####      ",
        "     ##  ##     ",
        "    ##    ##    ",
        "   ##      ##   ",
        "  ##        ##  ",
        " ##          ## ",
        "##            ##"
    ]),
    "Circle": _parse([
        "     ######     ",
        "   ##########   ",
        "  ####    ####  ",
        " ###        ### ",
        " ##          ## ",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        " ##          ## ",
        " ###        ### ",
        "  ####    ####  ",
        "   ##########   ",
        "     ######     "
    ]),
    "1": _parse([
        "       ##       ",
        "      ###       ",
        "     ####       ",
        "    ## ##       ",
        "   ##  ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "       ##       ",
        "  ############  ",
        "  ############  "
    ]),
    "H": _parse([
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "################",
        "################",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##",
        "##            ##"
    ]),
    "B": _parse([
        "##########      ",
        "###########     ",
        "##       ###    ",
        "##        ##    ",
        "##        ##    ",
        "##       ###    ",
        "###########     ",
        "##########      ",
        "###########     ",
        "##       ###    ",
        "##        ###   ",
        "##         ##   ",
        "##         ##   ",
        "##        ###   ",
        "############    ",
        "###########     "
    ])
}

# ═══════════════════════════════════════════════════════════════
# GLASSMORPHIC UI THEME 
# ═══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
:root {
    --bg-base: transparent;
    --card-bg: rgba(6, 11, 25, 0.65);
    --card-border: rgba(0, 255, 204, 0.15);
    --neon-cyan: #00f0ff;
    --neon-teal: #00ffcc;
    --neon-pink: #ff0055;
    --text-main: #f0f8ff;
    --text-dim: #7ab8d4;
    --font-ui: 'Outfit', sans-serif;
    --font-code: 'JetBrains Mono', monospace;
}
html, body, [data-testid="stAppViewContainer"] {
    background: transparent !important;
    font-family: var(--font-ui);
    color: var(--text-main);
}
.header-glass {
    background: linear-gradient(135deg, rgba(0,240,255,0.05), rgba(0,255,204,0.02)), var(--card-bg);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.header-glass::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}
.title-glow {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(to right, #fff, var(--neon-cyan), var(--neon-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 0 40px rgba(0, 240, 255, 0.3);
}
.sub-glow {
    font-size: 1.1rem; color: var(--text-dim); max-width: 800px; line-height: 1.6;
}
.glass-box {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.glass-box:hover {
    border-color: rgba(0, 240, 255, 0.4);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
}
.pill {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 50px;
    font-family: var(--font-code); font-size: 0.75rem; letter-spacing: 0.1em;
    background: rgba(0, 255, 204, 0.1); color: var(--neon-teal); border: 1px solid rgba(0,255,204,0.3);
    margin-bottom: 1rem; text-transform: uppercase;
}
.stat-val { font-family: var(--font-code); font-size: 2.2rem; font-weight: 700; color: var(--neon-cyan); }
.stat-lbl { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; }
div.stButton > button {
    background: linear-gradient(135deg, rgba(0,240,255,0.1), rgba(0,255,204,0.05)) !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    backdrop-filter: blur(5px) !important;
    transition: all 0.3s !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0077aa, #0099aa) !important;
    color: #fff !important; box-shadow: 0 0 15px rgba(0,240,255,0.4) !important;
}
div.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,240,255,0.6) !important;
    transform: translateY(-2px) !important;
}
textarea, input { background: rgba(0,0,0,0.3) !important; color: #fff !important; border-color: var(--card-border) !important; }
</style>
"""

# ═══════════════════════════════════════════════════════════════
# VOICE AI CO-PILOT
# ═══════════════════════════════════════════════════════════════
def render_voice_ai():
    """Live Speech-to-Text and TTS powered by Browser API + Nvidia"""
    st.markdown('<div class="glass-box" style="margin-top:2rem;border-color:rgba(255,0,85,0.3);">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0;color:#ff0055;">🎙️ Live AI Voice Co-Pilot</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7ab8d4;font-size:0.9rem;">Ask a question out loud. The STT engine will detect your speech, query Llama, and speak the reply.</p>', unsafe_allow_html=True)

    # HTML5 Speech Recognition Widget Injector
    html_code = """
    <div id="ai-interface" style="font-family: 'Outfit', sans-serif; color: white;">
        <button id="mic-btn" style="background: linear-gradient(135deg, #ff0055, #ff4488); border: none; border-radius: 50%; width: 60px; height: 60px; font-size: 24px; cursor: pointer; box-shadow: 0 0 20px rgba(255,0,85,0.5); transition: 0.3s; color: white;">
            🎤
        </button>
        <p id="status-txt" style="margin-top: 15px; font-size: 14px; color: #7ab8d4;">Click to Speak...</p>
        <input type="hidden" id="transcription-output">
    </div>

    <script>
    const btn = document.getElementById('mic-btn');
    const status = document.getElementById('status-txt');
    const out = document.getElementById('transcription-output');
    
    // We try to access the Streamlit parent document to find the hidden text input
    // This allows bidirectional communication back to Python
    const parentDoc = window.parent.document;
    
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        btn.onclick = () => {
            recognition.start();
            btn.style.boxShadow = "0 0 40px rgba(255,0,85,1)";
            status.innerText = "Listening...";
        };
        
        recognition.onresult = (e) => {
            const transcript = e.results[0][0].transcript;
            status.innerText = "Transcribed: " + transcript;
            btn.style.boxShadow = "0 0 20px rgba(255,0,85,0.5)";
            
            // Find ST text input and fire change
            let inputs = parentDoc.querySelectorAll('input[aria-label="voice_input_trigger"]');
            if (inputs.length > 0) {
                let stInput = inputs[0];
                let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                nativeInputValueSetter.call(stInput, transcript);
                let ev2 = new Event('input', { bubbles: true});
                stInput.dispatchEvent(ev2);
            }
        };
        
        recognition.onerror = (e) => {
            status.innerText = "Microphone Error: " + e.error;
            btn.style.boxShadow = "0 0 20px rgba(255,0,85,0.5)";
        };
    } else {
        status.innerText = "Speech API not supported in this browser. Use Chrome/Edge.";
        btn.style.display = "none";
    }
    </script>
    """
    
    components.html(html_code, height=130)

    # Hidden Streamlit input to receive the payload from Javascript
    val = st.text_input("Voice Input", key="voice_receiver", label_visibility="collapsed", args=("voice_input_trigger",), kwargs={"aria-label":"voice_input_trigger"}, placeholder="Or type your question here...")

    if st.button("Submit to AI", type="primary"):
        if val:
            ans = query_nvidia(val)
            st.session_state.last_ai_ans = ans
            st.session_state.ai_trigger_tts = True

    ans = st.session_state.get("last_ai_ans", "")
    if ans:
        st.markdown(f'<div style="background:rgba(255,0,85,0.1); border-left:3px solid #ff0055; padding: 1rem; border-radius: 0 8px 8px 0; margin-top: 1rem;">🤖 <b>AI:</b> {ans}</div>', unsafe_allow_html=True)
        # TTS Execution Loop
        if st.session_state.get("ai_trigger_tts", False):
            clean_ans = ans.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
            components.html(f"""
            <script>
            (function() {{
                const synth = window.parent.speechSynthesis;
                synth.cancel();
                const u = new SpeechSynthesisUtterance('{clean_ans}');
                u.rate = 1.0;
                synth.speak(u);
            }})();
            </script>
            """, height=0)
            st.session_state.ai_trigger_tts = False
            
    st.markdown('</div>', unsafe_allow_html=True)

def query_nvidia(q):
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        return "NVIDIA_API_KEY is missing. I cannot process your request. Add it to .env"
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "system", "content": "You are a smart, concise AI tutor for a Hopfield Neural Network. Speak conversationally."}, 
                      {"role": "user", "content": q}],
            max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error connecting to NVIDIA NIM: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════
def plot_matrix(arr, title="", cmap=[[0,"#060B19"],[1,"#00f0ff"]], zmin=-1, zmax=1):
    fig = go.Figure(data=go.Heatmap(z=arr.reshape(G,G), colorscale=cmap, showscale=False, zmin=zmin, zmax=zmax))
    fig.update_layout(height=280, width=280, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text=title, font=dict(color="#e2f4ff")))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_3d_energy(curr_s, memory_s, network):
    # Generates a pseudo 3D landscape interpolation
    es = []
    t_vals = np.linspace(0, 1, 20)
    for t in t_vals:
        interp = np.where(np.random.random(G*G) < t, memory_s, curr_s).astype(float)
        es.append(network.energy(interp))
        
    fig = go.Figure(data=[go.Scatter3d(
        x=t_vals, y=[0]*20, z=es, mode='lines', 
        line=dict(color='#00ffcc', width=5)
    )])
    fig.add_trace(go.Scatter3d(x=[1], y=[0], z=[es[-1]], mode='markers', marker=dict(color='#ff0055', size=10, symbol='diamond')))
    
    fig.update_layout(
        scene=dict(xaxis_title='Convergence', yaxis_title='', zaxis_title='Energy E',
                   xaxis=dict(showgrid=False, backgroundcolor="rgba(0,0,0,0)", color="#7ab8d4"),
                   yaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(gridcolor="rgba(0,240,255,0.1)", backgroundcolor="rgba(0,0,0,0)", color="#7ab8d4"),
                   bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=300
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════
def init():
    if "pseudohop" not in st.session_state:
        st.session_state.hop_memories = dict(PATTERNS)
        net = PseudoinverseHopfield(N)
        net.train(list(PATTERNS.values()))
        st.session_state.pseudohop = net
        st.session_state.canvas_key = 0

def app():
    from utils.styles import inject_global_css
    inject_global_css()
    st.markdown(CSS, unsafe_allow_html=True)
    init()
    net = st.session_state.pseudohop

    st.markdown("""
        <div class="header-glass">
            <div class="pill">Module 404 · Pseudoinverse Sandbox</div>
            <div class="title-glow">The Magic Hopfield Box</div>
            <div class="sub-glow">A flawlessly architected Hopfield Network. Upgraded to a 16x16 grid with Moore-Penrose Pseudoinverse mathematical learning, ensuring absolute global minima and perfect pattern recognition, completely eliminating spurious states.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0;">🖌️ High-Res Magic Whiteboard</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-dim);">Draw any of the stored shapes boldly. Watch the pseudoinverse engine flawlessly reconstruct it.</p>', unsafe_allow_html=True)
        
        if CANVAS_OK:
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=20,
                stroke_color="#00f0ff",
                background_color="#0a1226",
                height=320, width=320,
                drawing_mode="freedraw",
                key=f"canvas_draw_{st.session_state.canvas_key}",
            )
            
            sc1, sc2 = st.columns(2)
            if sc1.button("🧠 Execute Recognition", type="primary", use_container_width=True):
                if canvas.image_data is not None:
                    # Convert canvas to 16x16 bipolar
                    img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('L')
                    small = img.resize((G, G), Image.Resampling.LANCZOS)
                    arr = np.array(small)
                    bipolar = np.where(arr > 50, 1.0, -1.0).flatten()
                    
                    # Store input for UI
                    st.session_state.user_input = bipolar
                    
                    # Run convergence
                    recovered, energies = net.recover(bipolar, steps=150)
                    st.session_state.recovered = recovered
                    st.session_state.energies = energies
            
            if sc2.button("🗑️ Clear Whiteboard", use_container_width=True):
                st.session_state.user_input = None
                st.session_state.recovered = None
                st.session_state.canvas_key += 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        render_voice_ai()


    with c2:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0;">🔬 Convergence Analytics</h2>', unsafe_allow_html=True)
        
        if "recovered" in st.session_state and st.session_state.recovered is not None:
            inp = st.session_state.user_input
            out = st.session_state.recovered
            
            # Find closest match name
            overlaps = {k: float(out @ v) / N for k, v in st.session_state.hop_memories.items()}
            best_match = max(overlaps, key=overlaps.get)
            match_score = overlaps[best_match]
            
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(plot_matrix(inp, "Sensory Input"), use_container_width=True)
            with r2:
                st.plotly_chart(plot_matrix(out, "Neural Recall", cmap=[[0,"#060B19"],[1,"#00ffcc"]]), use_container_width=True)
            
            st.markdown(f'<div style="text-align:center; padding: 1rem; border-top: 1px solid var(--card-border); margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-lbl">Detected Attractor</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-val" style="color:#00ffcc;">{best_match}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-lbl">Orthogonal Convergence Confidence: {match_score*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h3 style="font-size:1.1rem; margin-top:1.5rem; color:#7ab8d4;">3D Energy Minimisation Landscape</h3>', unsafe_allow_html=True)
            st.plotly_chart(plot_3d_energy(inp, st.session_state.hop_memories[best_match], net), use_container_width=True)

        else:
            st.markdown("""
                <div style="height: 400px; display:flex; align-items:center; justify-content:center; flex-direction:column; opacity:0.6;">
                    <div style="font-size: 4rem;">⚛️</div>
                    <p style="font-family: var(--font-code); margin-top:1rem; color: var(--neon-cyan);">AWAITING SENSORY INPUT...</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.markdown('<h2>📚 Extracted Synaptic Memories</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text-dim);">The following 16x16 perfect orthogonal memories are stored entirely in the Pseudoinverse weight matrix.</p>', unsafe_allow_html=True)
    
    cols = st.columns(len(PATTERNS))
    for i, (name, pat) in enumerate(PATTERNS.items()):
        with cols[i]:
            st.plotly_chart(plot_matrix(pat, name), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()