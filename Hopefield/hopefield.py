"""
🧠 Live AI-Powered Hopfield Drawing Lab
Detects what you draw in real-time using NVIDIA Vision AI.
Auto-triggers on every canvas stroke.
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
        return st.secrets.get("NVIDIA_API_KEY", None)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════
# HOPFIELD ENGINE (educational visualization)
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
    img = Image.fromarray(data.astype('uint8'), 'RGBA').convert('L')
    img = img.filter(ImageFilter.BoxBlur(2))
    small = img.resize((G, G), Image.Resampling.LANCZOS)
    return np.where(np.array(small) > 15, 1.0, -1.0).flatten()

def canvas_to_base64(data):
    """Convert canvas to clean black-on-white PNG for best AI recognition."""
    img = Image.fromarray(data.astype('uint8'), 'RGBA')
    gray = img.convert('L')
    arr = np.array(gray)
    # Threshold: anything drawn becomes black, background becomes white
    bw = np.where(arr > 20, 0, 255).astype('uint8')
    # Make it thicker with a slight blur then re-threshold
    bw_img = Image.fromarray(bw, 'L').filter(ImageFilter.BoxBlur(2))
    bw_arr = np.array(bw_img)
    final = np.where(bw_arr < 200, 0, 255).astype('uint8')
    clean = Image.fromarray(final, 'L').convert('RGB')
    # Scale up to 256x256 for better AI visibility
    clean = clean.resize((256, 256), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    clean.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def canvas_hash(data):
    """Generate a hash of the canvas to detect changes."""
    gray = Image.fromarray(data.astype('uint8'), 'RGBA').convert('L')
    arr = np.array(gray)
    return hashlib.md5(arr.tobytes()).hexdigest()

def is_blank(data):
    gray = Image.fromarray(data.astype('uint8'), 'RGBA').convert('L')
    return np.array(gray).max() < 20

# ═══════════════════════════════════════════════════════════════
# NVIDIA VISION AI — LIVE DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_with_ai(canvas_data):
    key = get_api_key()
    if not key or not OPENAI_OK:
        return "?", "API key not loaded. Check your .env file."

    b64 = canvas_to_base64(canvas_data)

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
            max_tokens=100
        )
        result = resp.choices[0].message.content.strip()
        detected = "Drawing"
        for line in result.split('\n'):
            ln = line.strip()
            if ln.upper().startswith("NAME:"):
                detected = ln.split(":", 1)[1].strip().strip('"\' ')
                break
        # If the model didn't follow format, use the first word
        if detected == "Drawing" and result:
            detected = result.split()[0].strip(".,!:;\"'*#")
        return detected, result
    except Exception as e:
        # Fallback to text model
        try:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
            resp = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for a Hopfield Neural Network educational platform."},
                    {"role": "user", "content": "Explain in 2 sentences how a Hopfield network processes noisy drawings through energy minimization to reconstruct stored patterns."}
                ],
                temperature=0.7, max_tokens=120
            )
            return "Pattern", resp.choices[0].message.content
        except Exception as e2:
            return "?", f"Error: {e2}"

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_grid(arr, title=""):
    fig = go.Figure(data=go.Heatmap(
        z=arr.reshape(G, G), colorscale=[[0, "#0f172a"], [1, "#00ffcc"]],
        showscale=False, zmin=-1, zmax=1
    ))
    fig.update_layout(
        height=250, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(color="#E2E8F0", size=13))
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_energy(energies):
    fig = go.Figure(go.Scatter(
        y=energies, mode='lines+markers',
        line=dict(color='#8B5CF6', width=3),
        marker=dict(color='#3B82F6', size=5),
        fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'
    ))
    fig.update_layout(
        title=dict(text="Energy Minimisation", font=dict(color="#E2E8F0")),
        xaxis=dict(title="Step", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94A3B8"),
        yaxis=dict(title="E", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94A3B8"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_weight(W):
    fig = go.Figure(data=go.Heatmap(
        z=W, colorscale=[[0, "#8B5CF6"], [0.5, "#0f172a"], [1, "#00f0ff"]], showscale=True
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Synaptic Weight Matrix (256×256)", font=dict(color="#E2E8F0"))
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange='reversed')
    return fig

# ═══════════════════════════════════════════════════════════════
# INIT
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

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    init()
    engine = st.session_state.hop_engine

    # ── PREMIUM TITLE ──
    st.markdown("""
    <div class="premium-card">
        <h1 style="color:white; margin-bottom:4px; font-family:'Montserrat',sans-serif; font-weight:800;">🧠 Hopfield Neural Drawing Lab</h1>
        <p style="color:#A0AEC0; font-size:1.05rem; margin:0;">Draw anything on the canvas — NVIDIA Vision AI identifies it <b>live</b> as you draw, while the Hopfield engine demonstrates associative memory reconstruction in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── LAYOUT ──
    col_draw, col_out = st.columns([1, 1.3])

    with col_draw:
        with st.container(border=True):
            st.subheader("🖌️ Draw Here — Live Detection")
            st.caption("Every stroke triggers AI analysis automatically!")
            if CANVAS_OK:
                canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)", stroke_width=22, stroke_color="#00f0ff",
                    background_color="#0f172a", height=340, width=340, drawing_mode="freedraw",
                    key=f"hc_{st.session_state.hop_ck}"
                )

                # LIVE: Auto-detect on every canvas change
                if canvas.image_data is not None and not is_blank(canvas.image_data):
                    current_hash = canvas_hash(canvas.image_data)
                    if current_hash != st.session_state.hop_last_hash:
                        st.session_state.hop_last_hash = current_hash

                        # Hopfield processing
                        bipolar = canvas_to_bipolar(canvas.image_data)
                        st.session_state.hop_bipolar = bipolar
                        engine.store(bipolar)
                        recovered, energies = engine.recover(bipolar, steps=100)
                        st.session_state.hop_recovered = recovered
                        st.session_state.hop_energies = energies

                        # NVIDIA Vision AI
                        with st.spinner("🔍 AI analyzing..."):
                            detected, ai_text = detect_with_ai(canvas.image_data)
                            st.session_state.hop_detected = detected
                            st.session_state.hop_ai_text = ai_text

                if st.button("🗑️ Clear Canvas", use_container_width=True):
                    st.session_state.hop_ck += 1
                    st.session_state.hop_last_hash = None
                    st.session_state.hop_bipolar = None
                    st.session_state.hop_recovered = None
                    st.session_state.hop_detected = None
                    st.session_state.hop_ai_text = None
                    st.session_state.hop_engine = HopfieldEngine(N)
                    st.rerun()
            else:
                st.error("Install `streamlit-drawable-canvas`.")

    with col_out:
        with st.container(border=True):
            st.subheader("⚡ Live AI Detection")
            if st.session_state.hop_detected is not None:
                st.markdown(f"""
                <div style="text-align:center; padding: 10px 0;">
                    <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.15em;">AI Detected</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#00ffcc; font-family:'Montserrat',sans-serif;">{st.session_state.hop_detected}</div>
                </div>
                """, unsafe_allow_html=True)

                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(plot_grid(st.session_state.hop_bipolar, "Neural Grid (16×16)"), use_container_width=True, key="pc_inp")
                with g2:
                    st.plotly_chart(plot_grid(st.session_state.hop_recovered, "Hopfield Output"), use_container_width=True, key="pc_out")

                st.plotly_chart(plot_energy(st.session_state.hop_energies), use_container_width=True, key="pc_eng")
            else:
                st.info("Start drawing on the canvas — AI detection begins automatically with each stroke!")

        with st.container(border=True):
            st.subheader("🤖 AI Analysis")
            if st.session_state.hop_ai_text:
                # Render in a controlled font size to prevent oversized headers
                safe_text = st.session_state.hop_ai_text.replace('#', '').strip()
                st.markdown(f'<div style="font-size:0.9rem; line-height:1.6; color:#E2E8F0;">{safe_text}</div>', unsafe_allow_html=True)
            else:
                st.caption("Live analysis appears here as you draw.")

    # ── ARCHITECTURE DROPDOWN ──
    with st.expander("🛠️ Internal Working Steps, Logic & Architecture"):
        st.markdown("""
### Two-Engine Architecture

1. **NVIDIA Vision AI (Live)** — Every stroke triggers: canvas → PNG → base64 → `llama-3.2-90b-vision-instruct` → identification
2. **Hopfield Network (Educational)** — Canvas → 16×16 binary grid → 256 neurons → energy minimization → reconstruction

### Hopfield Step-by-Step
1. Canvas image → downsample to 16×16 → bipolar vector (+1/−1)
2. Store via outer product: $W += \\frac{p \\cdot p^T}{N}$
3. Feed noisy input → async neuron updates → energy decreases monotonically
4. Converge to attractor state (nearest stored pattern)
        """)

        st.markdown("### Synaptic Weight Matrix")
        st.plotly_chart(plot_weight(engine.W), use_container_width=True, key="pc_wt")

        if st.session_state.hop_energies is not None:
            st.markdown("### Energy Convergence")
            st.plotly_chart(plot_energy(st.session_state.hop_energies), use_container_width=True, key="pc_eng2")
            st.markdown("### Output Tensor")
            st.code(np.array2string(st.session_state.hop_recovered, max_line_width=80, separator=', '), language="python")

if __name__ == "__main__":
    main()