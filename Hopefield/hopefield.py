"""
🧠 Unified Hopfield Memory Engine
Clean container-based layout (no orphaned HTML divs), dotenv loading, safe error handling.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageFilter
import os

# ── Load .env FIRST so os.getenv works ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
G = 16
N = G * G

def get_api_key():
    """Safely retrieve NVIDIA API key from .env or Streamlit secrets."""
    k = os.getenv("NVIDIA_API_KEY")
    if k and k.strip():
        return k.strip()
    try:
        k2 = st.secrets.get("NVIDIA_API_KEY", None)
        if k2:
            return k2
    except Exception:
        pass
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

    def recover(self, s, steps=200):
        curr = s.copy()
        energies = [self.energy(curr)]
        for _ in range(steps):
            idx = np.random.randint(0, self.N)
            curr[idx] = 1.0 if (self.W[idx] @ curr) >= 0 else -1.0
            e = self.energy(curr)
            if abs(e - energies[-1]) > 1e-10:
                energies.append(e)
            sync_next = np.where(self.W @ curr >= 0, 1.0, -1.0)
            if np.array_equal(curr, sync_next):
                break
        return curr, energies

# ═══════════════════════════════════════════════════════════════
# CANVAS PROCESSOR
# ═══════════════════════════════════════════════════════════════
def process_canvas(canvas_data):
    """Direct downsample with blur — never returns empty."""
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA').convert('L')
    img = img.filter(ImageFilter.BoxBlur(2))
    small = img.resize((G, G), Image.Resampling.LANCZOS)
    arr = np.array(small)
    return np.where(arr > 15, 1.0, -1.0).flatten()

# ═══════════════════════════════════════════════════════════════
# PRE-BAKED INITIAL MEMORIES
# ═══════════════════════════════════════════════════════════════
def _p(rows):
    return np.array([1.0 if c == '#' else -1.0 for r in rows for c in r])

INITIAL_MEMORIES = {
    "Square": _p([
        "                ", "  ############  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ##        ##  ", "  ##        ##  ",
        "  ##        ##  ", "  ##        ##  ", "  ############  ", "                "
    ]),
    "Cross": _p([
        "                ", "      ####      ", "      ####      ", "      ####      ",
        "      ####      ", "      ####      ", "  ############  ", "  ############  ",
        "  ############  ", "  ############  ", "      ####      ", "      ####      ",
        "      ####      ", "      ####      ", "      ####      ", "                "
    ]),
    "X-Shape": _p([
        "                ", " ##          ## ", "  ##        ##  ", "   ##      ##   ",
        "    ##    ##    ", "     ##  ##     ", "      ####      ", "       ##       ",
        "       ##       ", "      ####      ", "     ##  ##     ", "    ##    ##    ",
        "   ##      ##   ", "  ##        ##  ", " ##          ## ", "                "
    ]),
}

# ═══════════════════════════════════════════════════════════════
# AI ANALYST (NVIDIA NIM)
# ═══════════════════════════════════════════════════════════════
def get_ai_analysis(action_type="recognize", mem_name=""):
    key = get_api_key()
    if not key or not OPENAI_OK:
        return None  # Return None, not an error string. We handle display separately.

    out = st.session_state.get("recovered")
    mems = st.session_state.get("hop_memories", {})

    if action_type == "learn":
        sys_p = "You are a concise Neural AI. The user stored a new shape in a Pseudoinverse Hopfield Network. Write 2 short futuristic sentences about the updates."
        usr_p = f"The user trained a new pattern named '{mem_name}'. Total patterns now: {len(mems)}."
    else:
        if out is None or len(mems) == 0:
            return "Network is idle. Draw and click Recognize to trigger analysis."
        overlaps = {k: float(out @ v) / N for k, v in mems.items()}
        bm = max(overlaps, key=overlaps.get)
        ens = st.session_state.get("energies", [0, 0])
        sys_p = "You are an AI Network Analyst. Provide exactly 2 short professional sentences."
        usr_p = f"Matched: {bm}. Confidence: {overlaps[bm]*100:.1f}%. Energy: {ens[0]:.1f} → {ens[-1]:.1f}."

    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            temperature=0.7, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI unavailable: {e}"

# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
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
        marker=dict(color='#3B82F6', size=6),
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
        height=350, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Synaptic Weight Matrix (W)", font=dict(color="#E2E8F0"))
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange='reversed')
    return fig

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════
def init():
    if "hop_memories" not in st.session_state:
        st.session_state.hop_memories = dict(INITIAL_MEMORIES)
        net = PseudoinverseHopfield(N)
        net.train(list(INITIAL_MEMORIES.values()))
        st.session_state.hop_net = net
        st.session_state.hop_canvas_key = 0
        st.session_state.hop_input = None
        st.session_state.hop_output = None
        st.session_state.hop_energies = None
        st.session_state.hop_ai = None

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    init()
    net = st.session_state.hop_net

    # ── HEADER ──
    st.markdown("# 🧠 The Unified Memory Matrix")
    st.caption("A flawless, single-screen Hopfield Engine powered by the Moore-Penrose Pseudoinverse. Draw any shape to reconstruct it, or teach the network new concepts dynamically.")

    # ── TWO COLUMNS ──
    col_draw, col_result = st.columns([1, 1.2])

    with col_draw:
        with st.container(border=True):
            st.subheader("🖌️ Drawing Board")
            if CANVAS_OK:
                canvas = st_canvas(
                    fill_color="rgba(0,0,0,0)", stroke_width=22, stroke_color="#00f0ff",
                    background_color="#0f172a", height=320, width=320, drawing_mode="freedraw",
                    key=f"hc_{st.session_state.hop_canvas_key}"
                )

                if st.button("🧠 Recognize & Reconstruct", use_container_width=True, type="primary"):
                    if canvas.image_data is not None:
                        bipolar = process_canvas(canvas.image_data)
                        st.session_state.hop_input = bipolar
                        st.session_state.hop_output, st.session_state.hop_energies = net.recover(bipolar, steps=200)
                        st.session_state.hop_ai = get_ai_analysis("recognize")

                lc, rc = st.columns([2, 1])
                new_name = lc.text_input("Name", placeholder="Your custom shape name", label_visibility="collapsed")
                if rc.button("💾 Learn", use_container_width=True):
                    if canvas.image_data is not None and new_name.strip():
                        bipolar = process_canvas(canvas.image_data)
                        st.session_state.hop_memories[new_name.strip()] = bipolar
                        net.train(list(st.session_state.hop_memories.values()))
                        st.session_state.hop_ai = get_ai_analysis("learn", new_name)
                        st.success(f"'{new_name}' stored!")
                    else:
                        st.warning("Draw something and name it first.")

                if st.button("🗑️ Clear Canvas", use_container_width=True):
                    st.session_state.hop_canvas_key += 1
                    st.rerun()
            else:
                st.error("Install `streamlit-drawable-canvas` to enable the whiteboard.")

    with col_result:
        with st.container(border=True):
            st.subheader("👁️ Neural Output")
            if st.session_state.hop_output is not None:
                out = st.session_state.hop_output
                mems = st.session_state.hop_memories
                if len(mems) > 0:
                    overlaps = {k: float(out @ v) / N for k, v in mems.items()}
                    best = max(overlaps, key=overlaps.get)
                    conf = overlaps[best] * 100
                else:
                    best = "N/A"
                    conf = 0.0

                st.metric("Detected Pattern", best, f"{conf:.1f}% confidence")

                r1, r2 = st.columns(2)
                with r1:
                    st.plotly_chart(plot_grid(st.session_state.hop_input, "Your Drawing"), use_container_width=True)
                with r2:
                    st.plotly_chart(plot_grid(out, "Reconstructed"), use_container_width=True)

                st.plotly_chart(plot_energy(st.session_state.hop_energies), use_container_width=True)
            else:
                st.info("Draw on the board and click **Recognize & Reconstruct** to see the neural output here.")

        # AI Analyst
        with st.container(border=True):
            st.subheader("🤖 AI Network Analyst")
            ai_txt = st.session_state.get("hop_ai")
            if ai_txt:
                st.markdown(f"> {ai_txt}")
            else:
                key = get_api_key()
                if not key:
                    st.warning("NVIDIA API Key not found. Please add `NVIDIA_API_KEY=nvapi-xxxx` to your `.env` file (the file is currently empty).")
                else:
                    st.caption("AI analysis will appear here after you run a recognition.")

    # ── STORED MEMORIES ──
    with st.container(border=True):
        st.subheader("📚 Stored Memory Library")
        mems = st.session_state.hop_memories
        if len(mems) == 0:
            st.info("No memories stored. Draw a shape and click Learn to add one.")
        else:
            ncols = min(len(mems), 5)
            cols = st.columns(ncols)
            for i, (name, pat) in enumerate(mems.items()):
                with cols[i % ncols]:
                    st.plotly_chart(plot_grid(pat, name), use_container_width=True)

        if st.button("🚨 Purge All Memories"):
            st.session_state.hop_memories = {}
            net.train([])
            st.session_state.hop_output = None
            st.session_state.hop_input = None
            st.session_state.hop_ai = None
            st.rerun()

    # ── ARCHITECTURE EXPLAINER ──
    with st.expander("🛠️ Internal Working Steps, Logic & Architecture"):
        st.markdown("### 1. How a Hopfield Network Works")
        st.markdown("""
A Hopfield Network is a **recurrent neural network** that acts as an associative memory.
It stores patterns as stable points (attractors) in an energy landscape.

**Architecture:**
- **Neurons:** 256 fully-connected binary neurons (16×16 grid)
- **Learning Rule:** Moore-Penrose Pseudoinverse ($W = X \\cdot X^+$)
- **Recovery:** Asynchronous neuron updates until convergence to minimum energy

**Step-by-step process:**
1. You draw on the canvas → image is downsampled to a 16×16 binary grid (256 values of +1 or -1)
2. This noisy vector is fed into the network
3. Each neuron updates based on the weighted sum of all other neurons
4. The network converges to the closest stored pattern (energy minimum)
5. The reconstructed pattern is displayed
        """)

        st.markdown("### 2. Live Synaptic Weight Matrix")
        st.markdown("This 256×256 matrix shows the connection strengths between every pair of neurons.")
        st.plotly_chart(plot_weight(net.W), use_container_width=True)

        if st.session_state.hop_output is not None:
            st.markdown("### 3. Energy Descent Trace")
            st.markdown("Each step reduces the system's energy until it reaches a stable attractor state.")
            st.plotly_chart(plot_energy(st.session_state.hop_energies), use_container_width=True)

            st.markdown("### 4. Raw Output Vector")
            st.code(np.array2string(st.session_state.hop_output, max_line_width=80, separator=', '), language="python")

if __name__ == "__main__":
    main()