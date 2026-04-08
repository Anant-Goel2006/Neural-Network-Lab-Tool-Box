"""
🧠 Magic Hopfield Lab - NeuralNetworkLab Component
Uses proper premium-card global theming & Pseudoinverse learning with auto-crop for perfect detection.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os, json
from openai import OpenAI
import time

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & SETUP
# ═══════════════════════════════════════════════════════════════
G = 16 
N = G * G

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
            # Async update random neuron
            idx = np.random.randint(0, self.N)
            curr[idx] = 1.0 if (self.W[idx] @ curr) >= 0 else -1.0
            e = self.energy(curr)
            
            # Avoid duplicate identical energy recordings back to back to keep charts clean
            if e != energies[-1]:
                energies.append(e)
            
            sync_next = np.where(self.W @ curr >= 0, 1.0, -1.0)
            if np.array_equal(curr, sync_next):
                break
        return curr, energies

# ═══════════════════════════════════════════════════════════════
# HIGH RES PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════
def _parse(str_arr):
    vals = []
    for row in str_arr:
        for char in row:
            vals.append(1.0 if char == '#' else -1.0)
    return np.array(vals)

PATTERNS = {
    "A": _parse([
        "      ####      ", "     ######     ", "    ##    ##    ", "   ##      ##   ",
        "   ##      ##   ", "  ##        ##  ", "  ##        ##  ", "  ############  ",
        "  ############  ", "  ##        ##  ", " ##          ## ", " ##          ## ",
        " ##          ## ", " ##          ## ", "##            ##", "##            ##"
    ]),
    "X": _parse([
        "##            ##", " ##          ## ", "  ##        ##  ", "   ##      ##   ",
        "    ##    ##    ", "     ##  ##     ", "      ####      ", "       ##       ",
        "       ##       ", "      ####      ", "     ##  ##     ", "    ##    ##    ",
        "   ##      ##   ", "  ##        ##  ", " ##          ## ", "##            ##"
    ]),
    "Circle": _parse([
        "     ######     ", "   ##########   ", "  ####    ####  ", " ###        ### ",
        " ##          ## ", "##            ##", "##            ##", "##            ##",
        "##            ##", "##            ##", "##            ##", " ##          ## ",
        " ###        ### ", "  ####    ####  ", "   ##########   ", "     ######     "
    ]),
    "1": _parse([
        "       ##       ", "      ###       ", "     ####       ", "    ## ##       ",
        "   ##  ##       ", "       ##       ", "       ##       ", "       ##       ",
        "       ##       ", "       ##       ", "       ##       ", "       ##       ",
        "       ##       ", "       ##       ", "  ############  ", "  ############  "
    ]),
    "H": _parse([
        "##            ##", "##            ##", "##            ##", "##            ##",
        "##            ##", "##            ##", "##            ##", "################",
        "################", "##            ##", "##            ##", "##            ##",
        "##            ##", "##            ##", "##            ##", "##            ##"
    ])
}

# ═══════════════════════════════════════════════════════════════
# AUTO-CROP ALGORITHM FOR HIGH ACCURACY
# ═══════════════════════════════════════════════════════════════
def process_canvas(canvas_data):
    """Auto-crops the drawn user strokes, centers them, and drops them perfectly into 16x16."""
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA')
    gray = img.convert('L')
    arr = np.array(gray)
    
    coords = np.column_stack(np.where(arr > 30))
    if coords.size == 0:
        return np.full(N, -1.0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped = gray.crop((x_min, y_min, x_max, y_max))
    w, h = cropped.size
    size = max(w, h)
    
    pad = int(size * 0.15) # 15% padding
    new_size = size + 2*pad
    
    square = Image.new('L', (new_size, new_size), color=0)
    paste_x = pad + (size - w) // 2
    paste_y = pad + (size - h) // 2
    square.paste(cropped, (paste_x, paste_y))
    
    small = square.resize((G, G), Image.Resampling.LANCZOS)
    return np.where(np.array(small) > 50, 1.0, -1.0).flatten()

# ═══════════════════════════════════════════════════════════════
# LIVE AI NETWORK ANALYST (NVIDIA NIM)
# ═══════════════════════════════════════════════════════════════
def analyze_network_state_ai():
    key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY", "")
    if not key:
        return "⚠️ NVIDIA API Key missing. Please provide the key in .env to activate the Live Network Analyst."
    
    # Build context from session
    inp = st.session_state.get("user_input")
    out = st.session_state.get("recovered")
    ens = st.session_state.get("energies", [])
    if out is None:
        return "The network is dormant, waiting for sensory input from the canvas. Draw a shape to trigger convergence!"
    
    overlaps = {k: float(out @ v) / N for k, v in PATTERNS.items()}
    bm = max(overlaps, key=overlaps.get)
    sc = overlaps[bm] * 100
    noise = int(((inp != out).sum() / N) * 100)
    
    sys_prompt = "You are the advanced 'AI Network Analyst' monitoring a 256-neuron Hopfield Network. The user just drew something, and the neural network attempted to recognize it. Give exactly 2 short, highly professional, sci-fi sounding analytical sentences breaking down this specific mathematical result. Do NOT use emojis."
    user_prompt = f"Data Dump:\n- Matched Entity: {bm}\n- Confidence: {sc:.1f}%\n- Energy Dropped from {ens[0]:.1f} to {ens[-1]:.1f}\n- Initial Noise Overcome: {noise}%"
    
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Neural interface interrupted: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════
def plot_matrix(arr, title=""):
    fig = go.Figure(data=go.Heatmap(z=arr.reshape(G,G), colorscale=[[0, "rgba(59, 130, 246, 0.05)"], [1, "#00ffcc"]], showscale=False, zmin=-1, zmax=1))
    fig.update_layout(height=250, width=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text=title, font=dict(color="#E2E8F0")))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_energy(energies):
    fig = go.Figure(go.Scatter(y=energies, mode='lines+markers', line=dict(color='#8B5CF6', width=3), marker=dict(color='#3B82F6', size=8), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
    fig.update_layout(
        title=dict(text="Energy Minimisation Trajectory", font=dict(color="#E2E8F0")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0,r=0,t=40,b=0)
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    
    if "hop_w" not in st.session_state:
        app_net = PseudoinverseHopfield(N)
        app_net.train(list(PATTERNS.values()))
        st.session_state.hop_w = app_net
        st.session_state.canvas_key = 0

    net = st.session_state.hop_w

    st.markdown("""
    <div class="premium-card">
        <h1 style="color:white; margin-bottom:0;">🧠 Neural Associative Memory</h1>
        <p style="color:#A0AEC0; font-size:1.1rem;">Driven by a strict 256-Neuron Moore-Penrose Pseudoinverse Engine, ensuring mathematically perfect memory retrieval.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="premium-card neon-pulse-border">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white; margin-top:0;">🖌️ Detection Whiteboard</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:#A0AEC0; font-size:0.9rem;">Draw a letter like A, X, H, 1, or a Circle. Our Auto-Crop AI will perfectly center it for neural processing.</p>', unsafe_allow_html=True)
        
        if CANVAS_OK:
            canvas = st_canvas(
                fill_color="rgba(0,0,0,0)", stroke_width=25, stroke_color="#00ffcc",
                background_color="#0f172a", height=300, width=300, drawing_mode="freedraw",
                key=f"hw_{st.session_state.canvas_key}"
            )
            
            c_a, c_b = st.columns(2)
            if c_a.button("🧠 Execute Recall", use_container_width=True, type="primary"):
                if canvas.image_data is not None:
                    # Execute high precision crop
                    bipolar = process_canvas(canvas.image_data)
                    st.session_state.user_input = bipolar
                    st.session_state.recovered, st.session_state.energies = net.recover(bipolar, steps=150)
                    
                    # Store AI Analyst response statefully to prevent over-fetching
                    st.session_state.ai_analysis = analyze_network_state_ai()
            
            if c_b.button("🗑️ Clear", use_container_width=True):
                st.session_state.user_input = None
                st.session_state.recovered = None
                st.session_state.ai_analysis = None
                st.session_state.canvas_key += 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stored Memories
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:white; margin-top:0;">📚 Perfect Orthogonal Memories</h4>', unsafe_allow_html=True)
        cols = st.columns(len(PATTERNS))
        for i, (name, pat) in enumerate(PATTERNS.items()):
            with cols[i]:
                st.plotly_chart(plot_matrix(pat, name), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white; margin-top:0;">🤖 Live NVIDIA Network Analyst</h3>', unsafe_allow_html=True)
        analyst_text = st.session_state.get("ai_analysis", analyze_network_state_ai())
        st.markdown(f'<div style="background:rgba(59, 130, 246, 0.1); border-left:4px solid #3B82F6; padding:15px; border-radius:4px; color:#E2E8F0; font-family:\'JetBrains Mono\', monospace; font-size:0.9rem;">{analyst_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white; margin-top:0;">🔬 Telemetry</h3>', unsafe_allow_html=True)
        
        if st.session_state.get("recovered") is not None:
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(plot_matrix(st.session_state.user_input, "Auto-Cropped Input"), use_container_width=True)
            with r2:
                st.plotly_chart(plot_matrix(st.session_state.recovered, "Neural Convergence"), use_container_width=True)
            
            st.plotly_chart(plot_energy(st.session_state.energies), use_container_width=True)
        else:
            st.markdown('<div style="opacity:0.3; text-align:center; padding: 40px; font-family:\'JetBrains Mono\';">Waiting for sensory input matrix...</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()