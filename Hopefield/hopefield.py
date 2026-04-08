"""
🧠 Unified Hopfield Memory Engine 
Zero "empty box" architecture with direct canvas translation and live AI evaluation.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os
from openai import OpenAI

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & SAFE SECRETS
# ═══════════════════════════════════════════════════════════════
G = 16 
N = G * G

def get_api_key():
    k = os.getenv("NVIDIA_API_KEY")
    if k: return k
    try:
        from streamlit import secrets
        if "NVIDIA_API_KEY" in secrets:
            return secrets["NVIDIA_API_KEY"]
    except Exception: pass
    return None

# ═══════════════════════════════════════════════════════════════
# PSEUDOINVERSE HOPFIELD ENGINE
# ═══════════════════════════════════════════════════════════════
class PseudoinverseHopfield:
    def __init__(self, size=256):
        self.N = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        if not patterns:
            self.W = np.zeros((self.N, self.N))
            return
        X = np.column_stack(patterns)
        try:
            X_inv = np.linalg.pinv(X)
            W = X @ X_inv
            np.fill_diagonal(W, 0)
            self.W = W
        except Exception:
            self.W = np.zeros((self.N, self.N))

    def energy(self, s):
        return -0.5 * float(s @ self.W @ s)

    def recover(self, s, steps=150):
        curr = s.copy()
        energies = [self.energy(curr)]
        for _ in range(steps):
            idx = np.random.randint(0, self.N)
            curr[idx] = 1.0 if (self.W[idx] @ curr) >= 0 else -1.0
            e = self.energy(curr)
            if e != energies[-1]:
                energies.append(e)
            
            sync_next = np.where(self.W @ curr >= 0, 1.0, -1.0)
            if np.array_equal(curr, sync_next):
                break
        return curr, energies

# ═══════════════════════════════════════════════════════════════
# DIRECT PIXEL TRANSLATION
# ═══════════════════════════════════════════════════════════════
def process_canvas(canvas_data):
    """Bulletproof translation: direct box-blurred downsampling ensures NO empty outputs."""
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA').convert('L')
    # Soften hard lines before downsampling
    img = img.filter(ImageFilter.BoxBlur(1))
    small = img.resize((G, G), Image.Resampling.LANCZOS)
    arr = np.array(small)
    return np.where(arr > 15, 1.0, -1.0).flatten()

# ═══════════════════════════════════════════════════════════════
# PRE-BAKED INITIAL MEMORIES
# ═══════════════════════════════════════════════════════════════
def _parse(str_arr):
    return np.array([1.0 if c == '#' else -1.0 for r in str_arr for c in r])

INITIAL_MEMORIES = {
    "Square": _parse([
        "                ", "  ############  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ############  ", "                "
    ]),
    "Cross": _parse([
        "                ", "      ####      ", "      ####      ", "      ####      ",
        "      ####      ", "      ####      ", "  ############  ", "  ############  ",
        "  ############  ", "  ############  ", "      ####      ", "      ####      ",
        "      ####      ", "      ####      ", "      ####      ", "                "
    ]),
    "X-Mark": _parse([
        "                ", " ##          ## ", "  ##        ##  ", "   ##      ##   ",
        "    ##    ##    ", "     ##  ##     ", "      ####      ", "       ##       ",
        "       ##       ", "      ####      ", "     ##  ##     ", "    ##    ##    ",
        "   ##      ##   ", "  ##        ##  ", " ##          ## ", "                "
    ])
}

# ═══════════════════════════════════════════════════════════════
# LIVE AI NETWORK ANALYST (NVIDIA NIM)
# ═══════════════════════════════════════════════════════════════
def get_ai_analysis(action_type, mem_name=""):
    key = get_api_key()
    if not key:
        return "⚠️ NVIDIA API Key missing entirely. Please populate `.env` with NVIDIA_API_KEY. Operations will continue locally without AI descriptions."
    
    out = st.session_state.get("recovered")
    if action_type == "learn":
        sys_prompt = "You are a concise Neural AI. The user just burned a new shape into a Pseudoinverse Hopfield Network."
        user_prompt = f"The user trained a new pattern named '{mem_name}'. Write 2 short, futuristic sentences explaining that the synaptic weight matrix was orthogonally updated."
    else:
        if out is None: return "Sensory input requested. Draw on the board and click Recognize."
        ens = st.session_state.get("energies", [0, 0])
        overlaps = {k: float(out @ v) / N for k, v in st.session_state.hop_memories.items()}
        bm = max(overlaps, key=overlaps.get)
        
        sys_prompt = "You are an AI Network Analyst. Provide exactly 2 short, professional sentences analyzing the Hopfield memory recall."
        user_prompt = f"Matched Memory: {bm}. Confidence: {overlaps[bm]*100:.1f}%. Energy optimized from {ens[0]:.1f} to {ens[-1]:.1f}."
        
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Analyst unavailable: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════
def plot_matrix(arr, title=""):
    fig = go.Figure(data=go.Heatmap(z=arr.reshape(G,G), colorscale=[[0, "rgba(59, 130, 246, 0.05)"], [1, "#00ffcc"]], showscale=False, zmin=-1, zmax=1))
    fig.update_layout(height=240, width=240, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text=title, font=dict(color="#E2E8F0")))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_energy(energies):
    fig = go.Figure(go.Scatter(y=energies, mode='lines+markers', line=dict(color='#8B5CF6', width=3), marker=dict(color='#3B82F6', size=8), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
    fig.update_layout(
        title=dict(text="Energy Function Minimization", font=dict(color="#E2E8F0")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0,r=0,t=40,b=0)
    )
    return fig

def plot_weight_matrix(W):
    fig = go.Figure(data=go.Heatmap(z=W, colorscale=[[0, "#8B5CF6"], [0.5, "rgba(15, 23, 42, 0.8)"], [1, "#00f0ff"]], showscale=True))
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text="Live Synaptic Weight Matrix (W)", font=dict(color="#E2E8F0")))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

# ═══════════════════════════════════════════════════════════════
# STATE INITIALIZER
# ═══════════════════════════════════════════════════════════════
def init():
    if "hop_memories" not in st.session_state:
        st.session_state.hop_memories = dict(INITIAL_MEMORIES)
        net = PseudoinverseHopfield(N)
        net.train(list(INITIAL_MEMORIES.values()))
        st.session_state.hop_w = net
        st.session_state.canvas_key = 0
        st.session_state.user_input = None
        st.session_state.recovered = None
        st.session_state.energies = None
        st.session_state.ai_analysis = "Awaiting sensory input. Draw a shape to trigger analysis."

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    init()
    net = st.session_state.hop_w

    st.markdown("""
    <div class="premium-card">
        <h1 style="color:white; margin-bottom:0;">🧠 The Unified Memory Matrix</h1>
        <p style="color:#A0AEC0; font-size:1.1rem;">A flawless, single-screen Hopfield Engine. Draw any shape to perfectly reconstruct it, or dynamically teach the network entirely new concepts.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])

    # -----------------------------------------------------------
    # LEFT COLUMN: INTERFACE
    # -----------------------------------------------------------
    with c1:
        st.markdown('<div class="premium-card neon-pulse-border">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white; margin-top:0;">🖌️ Omnipresent Drawing Board</h3>', unsafe_allow_html=True)
        
        if CANVAS_OK:
            canvas = st_canvas(
                fill_color="rgba(0,0,0,0)", stroke_width=25, stroke_color="#00f0ff",
                background_color="#0f172a", height=320, width=320, drawing_mode="freedraw",
                key=f"hc_{st.session_state.canvas_key}"
            )
            
            # Action Row 1: RECOGNIZE
            if st.button("🧠 Recognize & Reconstruct", use_container_width=True, type="primary"):
                if canvas.image_data is not None:
                    bipolar = process_canvas(canvas.image_data)
                    st.session_state.user_input = bipolar
                    st.session_state.recovered, st.session_state.energies = net.recover(bipolar, steps=150)
                    st.session_state.ai_analysis = get_ai_analysis("recognize")
                    
            # Action Row 2: LEARN
            ca, cb = st.columns([2, 1])
            new_name = ca.text_input("Name", placeholder="Name your custom shape", label_visibility="collapsed")
            if cb.button("💾 Learn Shape", use_container_width=True):
                if canvas.image_data is not None and new_name.strip():
                    bipolar = process_canvas(canvas.image_data)
                    st.session_state.hop_memories[new_name.strip()] = bipolar
                    net.train(list(st.session_state.hop_memories.values()))
                    st.session_state.ai_analysis = get_ai_analysis("learn", new_name)
                    st.success(f"Perfectly added '{new_name}' to synaptic matrix!")
                else:
                    st.warning("Draw a shape and give it a name first!")
            
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.canvas_key += 1
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # RIGHT COLUMN: VISUAL OUTPUTS & AI
    # -----------------------------------------------------------
    with c2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white; margin-top:0;">👁️ Live Neural Perception Output</h3>', unsafe_allow_html=True)
        
        if st.session_state.recovered is not None:
            out = st.session_state.recovered
            overlaps = {k: float(out @ v) / N for k, v in st.session_state.hop_memories.items()}
            best_match = max(overlaps, key=overlaps.get)
            
            st.markdown(f'<div style="text-align:center; padding-bottom: 20px;"><div style="font-size:0.9rem; color:#A0AEC0; text-transform:uppercase; letter-spacing:0.1em;">Most Probable Concept</div><div style="font-size:2.5rem; font-weight:bold; color:#00ffcc; text-shadow:0 0 20px rgba(0,255,204,0.4);">{best_match}</div></div>', unsafe_allow_html=True)
            
            p1, p2 = st.columns(2)
            with p1: st.plotly_chart(plot_matrix(st.session_state.user_input, "What you drew"), use_container_width=True)
            with p2: st.plotly_chart(plot_matrix(out, "What it remembered"), use_container_width=True)
        else:
            st.markdown('<div style="opacity:0.4; text-align:center; padding: 60px; font-family:\'JetBrains Mono\';">Sensory array offline. Awaiting canvas signal...</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # NVIDIA LLaMA Output
        st.markdown(f'''
        <div class="premium-card" style="border-left: 4px solid #3B82F6;">
            <h4 style="color:#00f0ff; margin-top:0; font-family:'JetBrains Mono';">🤖 Live Analytical Readout</h4>
            <div style="color:#E2E8F0; font-size:0.95rem; line-height:1.6;">{st.session_state.ai_analysis}</div>
        </div>
        ''', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # LOGIC ARCHITECTURE (Expander)
    # -----------------------------------------------------------
    with st.expander("🛠️ View Internal Working Steps, Logic Architecture & Mathematics"):
        st.markdown('<div class="premium-card" style="background: rgba(15, 23, 42, 0.95);">', unsafe_allow_html=True)
        st.markdown("## 1. Pseudoinverse Synaptic Weight Matrix (W)")
        st.markdown("The complete associative memory topography. Each added concept mathematically warps this exact matrix.")
        st.plotly_chart(plot_weight_matrix(net.W), use_container_width=True)
        
        if st.session_state.recovered is not None:
            st.markdown("## 2. Mathematical Descent & Optimization")
            st.markdown("Upon attempting to 'Recognize', the noisy vector begins interacting with the weight matrix. The state rolls downwards until it hits a basin of attraction.")
            st.plotly_chart(plot_energy(st.session_state.energies), use_container_width=True)
            
            st.markdown("## 3. Reconstructed Output Tensor")
            st.code(np.array2string(st.session_state.recovered, max_line_width=80, separator=', '), language="python")
        else:
            st.markdown("## 2/3. Optimization Steps & Output")
            st.info("Execute 'Recognize' to generate live energy traces and tensor outputs.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # -----------------------------------------------------------
    # STORED MEMORIES LIBRARY
    # -----------------------------------------------------------
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:white; margin-top:0;">📚 Deep Structural Memories</h3>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (name, pat) in enumerate(st.session_state.hop_memories.items()):
        with cols[i % 6]: st.plotly_chart(plot_matrix(pat, name), use_container_width=True)
    
    if st.button("🚨 Purge All Synaptic Pathways (Empty Matrix)", type="primary"):
        st.session_state.hop_memories = {}
        net.train([])
        st.session_state.recovered = None
        st.session_state.ai_analysis = get_ai_analysis("purge")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()