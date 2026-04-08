"""
🧠 Live AI-Powered Hopfield Drawing Lab
Detects what you draw in real-time using NVIDIA Vision AI.
Auto-triggers on every canvas stroke. Full-width layout.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os, io, base64, hashlib

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

# ═══════════════════════════════════════════════════════════════
G = 16
N = G * G

def get_api_key():
    k = os.getenv("NVIDIA_API_KEY")
    if k and k.strip():
        return k.strip()
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════
# HOPFIELD ENGINE
# ═══════════════════════════════════════════════════════════════
class HopfieldEngine:
    def __init__(self, size=256):
        self.N = size
        self.W = np.zeros((size, size))

    def store(self, pattern):
        p = pattern.reshape(-1, 1)
        self.W += (p @ p.T) / self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return -0.5 * float(s @ self.W @ s)

    def recover(self, s, steps=100):
        curr = s.copy()
        energies = [self.energy(curr)]
        for _ in range(steps):
            i = np.random.randint(0, self.N)
            curr[i] = 1.0 if (self.W[i] @ curr) >= 0 else -1.0
            e = self.energy(curr)
            if abs(e - energies[-1]) > 1e-10:
                energies.append(e)
            if np.array_equal(curr, np.where(self.W @ curr >= 0, 1.0, -1.0)):
                break
        return curr, energies

# ═══════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════
def canvas_to_bipolar(data):
    if data is None or not isinstance(data, np.ndarray): return np.zeros(N)
    img = Image.fromarray(data.astype('uint8'), 'RGBA').convert('L')
    img = img.filter(ImageFilter.BoxBlur(2))
    small = img.resize((G, G), Image.Resampling.LANCZOS)
    return np.where(np.array(small) > 15, 1.0, -1.0).flatten()

def canvas_to_base64(data):
    """Convert canvas to clean black-on-white PNG for best AI recognition."""
    if data is None or not isinstance(data, np.ndarray): return ""
    try:
        img = Image.fromarray(data.astype('uint8'), 'RGBA')
        gray = img.convert('L')
        arr = np.array(gray)
        # Threshold: drawn is black, bg is white
        bw = np.where(arr > 20, 0, 255).astype('uint8')
        bw_img = Image.fromarray(bw, 'L').filter(ImageFilter.BoxBlur(2))
        final = np.where(np.array(bw_img) < 200, 0, 255).astype('uint8')
        clean = Image.fromarray(final, 'L').convert('RGB')
        clean = clean.resize((256, 256), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        clean.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return ""

def canvas_hash(data):
    if data is None or not isinstance(data, np.ndarray): return None
    try:
        return hashlib.md5(data.tobytes()).hexdigest()
    except Exception:
        return None

def is_blank(data):
    if data is None or not isinstance(data, np.ndarray): return True
    try:
        gray = Image.fromarray(data.astype('uint8'), 'RGBA').convert('L')
        return np.array(gray).max() < 20
    except Exception:
        return True

# ═══════════════════════════════════════════════════════════════
# NVIDIA VISION AI
# ═══════════════════════════════════════════════════════════════
def detect_with_ai(canvas_data):
    key = get_api_key()
    if not key or not OPENAI_OK:
        return "?", "API key not loaded. Check your .env file."

    b64 = canvas_to_base64(canvas_data)
    if not b64: return "?", "Canvas parsing error."

    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this hand-drawn black ink sketch on white paper. What single letter, number, or shape is drawn? Answer in this exact format on two lines:\nNAME: (one word only, like A or Circle or 7)\nNOTE: (one short sentence about the drawing)"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=60
        )
        result = resp.choices[0].message.content.strip()
        detected = "Drawing"
        for line in result.split('\n'):
            ln = line.strip()
            if ln.upper().startswith("NAME:"):
                detected = ln.split(":", 1)[1].strip().strip('"\' ')
                break
        if detected == "Drawing" and result:
             detected = result.split()[0].strip(".,!:;\"'*#\n")
        return detected[:20], result
    except Exception as e:
        return "?", f"API Error: {e}"

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_grid(arr, title=""):
    fig = go.Figure(data=go.Heatmap(
        z=arr.reshape(G, G), colorscale=[[0, "#0f172a"], [1, "#00ffcc"]],
        showscale=False, zmin=-1, zmax=1
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(color="#E2E8F0", size=13))
    )
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_energy(energies):
    fig = go.Figure(go.Scatter(
        y=energies, mode='lines+markers', line=dict(color='#8B5CF6', width=3),
        marker=dict(color='#3B82F6', size=5), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'
    ))
    fig.update_layout(
        title=dict(text="Energy Minimisation", font=dict(color="#E2E8F0")),
        xaxis=dict(title="Step", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94A3B8"),
        yaxis=dict(title="Energy", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94A3B8"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_weight(W):
    fig = go.Figure(data=go.Heatmap(z=W, colorscale=[[0, "#8B5CF6"], [0.5, "#0f172a"], [1, "#00f0ff"]], showscale=True))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Synaptic Weight Matrix (W)", font=dict(color="#E2E8F0"))
    )
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

# ═══════════════════════════════════════════════════════════════
# INIT & LOGIC
# ═══════════════════════════════════════════════════════════════
def init():
    if "hop_engine" not in st.session_state:
        st.session_state.hop_engine = HopfieldEngine(N)
        st.session_state.hop_ck = 0
        st.session_state.hop_last_hash = None
        st.session_state.hop_bipolar = None
        st.session_state.hop_recovered = None
        st.session_state.hop_energies = None
        st.session_state.hop_detected = None
        st.session_state.hop_ai_text = None

def clear_canvas():
    st.session_state.hop_ck += 1
    st.session_state.hop_last_hash = None
    st.session_state.hop_bipolar = None
    st.session_state.hop_recovered = None
    st.session_state.hop_detected = None
    st.session_state.hop_ai_text = None
    st.session_state.hop_engine = HopfieldEngine(N)

# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    init()
    engine = st.session_state.hop_engine

    st.markdown("""
    <div class="premium-card">
        <h1 style="color:white; margin-bottom:4px; font-family:'Montserrat',sans-serif; font-weight:800;">🧠 Full-Span Hopfield Neural Array</h1>
        <p style="color:#A0AEC0; font-size:1.05rem; margin:0;">Massive live canvas. Draw anything — AI detects it instantly below.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Full-width Canvas
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.subheader("🖌️ Draw Box")
            st.caption("Live AI tracking active.")
            if st.button("🗑️ Clear Canvas", use_container_width=True, type="primary"):
                clear_canvas()
                st.rerun()
                
        with col2:
            if CANVAS_OK:
                # Use large, rectangular canvas covering the screen width
                canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)", stroke_width=25, stroke_color="#00f0ff",
                    background_color="#0f172a", height=380, width=900, drawing_mode="freedraw",
                    key=f"hc_{st.session_state.hop_ck}"
                )

                # Process changes automatically
                if canvas.image_data is not None and not is_blank(canvas.image_data):
                    current_hash = canvas_hash(canvas.image_data)
                    if current_hash != st.session_state.hop_last_hash:
                        st.session_state.hop_last_hash = current_hash

                        bipolar = canvas_to_bipolar(canvas.image_data)
                        st.session_state.hop_bipolar = bipolar
                        engine.store(bipolar)
                        recovered, energies = engine.recover(bipolar, steps=100)
                        st.session_state.hop_recovered = recovered
                        st.session_state.hop_energies = energies

                        with st.spinner("🔍 Scanning neural structure..."):
                            detected, ai_text = detect_with_ai(canvas.image_data)
                            st.session_state.hop_detected = detected
                            st.session_state.hop_ai_text = ai_text
            else:
                st.error("Missing `streamlit-drawable-canvas`")

    # 2. Output Panel (Below Canvas)
    if st.session_state.hop_detected is not None:
        c_left, c_right = st.columns([1.2, 2])
        
        with c_left:
            with st.container(border=True):
                st.markdown(f"""
                <div style="text-align:center; padding: 20px 0;">
                    <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:10px;">⚡ AI Identification</div>
                    <div style="font-size:3.5rem; font-weight:800; color:#00ffcc; line-height:1.1; font-family:'Montserrat',sans-serif; text-shadow: 0 0 20px rgba(0,255,204,0.3);">{st.session_state.hop_detected}</div>
                </div>
                <hr style="border-top:1px solid rgba(255,255,255,0.1); margin:10px 0;">
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='font-size:0.85rem; color:#A0AEC0; margin-bottom:5px;'>🤖 Analysis:</div>", unsafe_allow_html=True)
                safe_text = st.session_state.hop_ai_text.replace('#', '').strip()
                st.markdown(f'<div style="font-size:1.1rem; line-height:1.6; color:#F8FAFC;">{safe_text}</div>', unsafe_allow_html=True)

        with c_right:
            with st.container(border=True):
                st.markdown("<h4 style='margin-top:0;'>Neural Topography & Energy Process</h4>", unsafe_allow_html=True)
                nc1, nc2, nc3 = st.columns(3)
                with nc1: st.plotly_chart(plot_grid(st.session_state.hop_bipolar, "Sensor (16x16)"), use_container_width=True, key="pg1")
                with nc2: st.plotly_chart(plot_grid(st.session_state.hop_recovered, "Network Output"), use_container_width=True, key="pg2")
                with nc3: st.plotly_chart(plot_weight(engine.W), use_container_width=True, key="pw1")
                st.plotly_chart(plot_energy(st.session_state.hop_energies), use_container_width=True, key="pe1")

if __name__ == "__main__":
    main()