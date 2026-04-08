"""
🧠 Custom Hopfield Memory Lab
Dual-Canvas system: Users teach custom shapes, then test the memory.
Includes safe secrets handling and transparent architectural steps.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
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
    # Safely get key without crashing Windows on missing secrets.toml
    k = os.getenv("NVIDIA_API_KEY")
    if k: return k
    try:
        from streamlit import secrets
        if "NVIDIA_API_KEY" in secrets:
            return secrets["NVIDIA_API_KEY"]
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
# AUTO-CROP ALGORITHM
# ═══════════════════════════════════════════════════════════════
def process_canvas(canvas_data):
    """Auto-crops the drawn user strokes and perfectly centers them into 16x16."""
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
    
    pad = int(size * 0.15)
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
    key = get_api_key()
    if not key:
        return "⚠️ NVIDIA API Key missing. Please provide the key in .env to activate the Live Network Analyst."
    
    if len(st.session_state.hop_memories) == 0:
        return "The network is an empty slate. Awaiting the user to forge new synaptic connections by storing custom shapes."
        
    out = st.session_state.get("recovered")
    ens = st.session_state.get("energies", [])
    if out is None:
        return f"The memory matrix holds {len(st.session_state.hop_memories)} unique conceptual diagrams. Awaiting corrupted sensory input to reconstruct."
    
    overlaps = {k: float(out @ v) / N for k, v in st.session_state.hop_memories.items()}
    if not overlaps: return "Analysis pending."
    bm = max(overlaps, key=overlaps.get)
    sc = overlaps[bm] * 100
    
    sys_prompt = "You are an 'AI Network Analyst'. The user trained a custom Hopfield associative memory. Give 2 highly professional, sci-fi analytical sentences breaking down the mathematical recovery result."
    user_prompt = f"Data:\n- Custom concept recovered: {bm}\n- Mathematical Confidence: {sc:.1f}%\n- Energy dropped from {ens[0]:.1f} to {ens[-1]:.1f}."
    
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
    fig.update_layout(height=280, width=280, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text=title, font=dict(color="#E2E8F0")))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange='reversed')
    return fig

def plot_energy(energies):
    fig = go.Figure(go.Scatter(y=energies, mode='lines+markers', line=dict(color='#8B5CF6', width=3), marker=dict(color='#3B82F6', size=8), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
    fig.update_layout(
        title=dict(text="Energy Function Minimization", font=dict(color="#E2E8F0")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0", title="Steps"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#E2E8F0", title="Energy E"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0,r=0,t=40,b=0)
    )
    return fig

def plot_weight_matrix(W):
    fig = go.Figure(data=go.Heatmap(z=W, colorscale=[[0, "#8B5CF6"], [0.5, "rgba(15, 23, 42, 0.8)"], [1, "#00ffcc"]], showscale=True))
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=dict(text="Live Synaptic Weight Matrix (W)", font=dict(color="#E2E8F0")))
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    from utils.styles import inject_global_css
    inject_global_css()
    
    if "hop_memories" not in st.session_state:
        st.session_state.hop_memories = {}
        st.session_state.hop_w = PseudoinverseHopfield(N)
        st.session_state.canvas_tst_key = 0
        st.session_state.canvas_trn_key = 1000

    net = st.session_state.hop_w

    st.markdown("""
    <div class="premium-card">
        <h1 style="color:white; margin-bottom:0;">🧠 Custom Hopfield Architecture</h1>
        <p style="color:#A0AEC0; font-size:1.1rem;">A true memory sandbox. Teach the network <b>any unique diagram</b>. Even if corrupted later, the mathematical matrix will cleanly reconstruct your exact drawing.</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["🖌️ Step 1: Teach the Network (Store Memory)", "🔍 Step 2: Test the Network (Recall)"])

    # -----------------------------------------------------------
    # TAB 1: TEACHING
    # -----------------------------------------------------------
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="premium-card neon-pulse-border">', unsafe_allow_html=True)
            st.markdown('<h3 style="color:white; margin-top:0;">1. The Memory Forge</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color:#A0AEC0; font-size:0.9rem;">Draw a unique shape (e.g. a Star, Smiley, or custom rune). It will be permanently burned into the neural matrix.</p>', unsafe_allow_html=True)
            
            if CANVAS_OK:
                canvas_train = st_canvas(
                    fill_color="rgba(0,0,0,0)", stroke_width=25, stroke_color="#00ffcc",
                    background_color="#0f172a", height=300, width=300, drawing_mode="freedraw",
                    key=f"hw_trn_{st.session_state.canvas_trn_key}"
                )
                
                t_name = st.text_input("Name this diagram:", placeholder="e.g. 'Star', 'Tree', 'Symbol_1'")
                
                ca, cb = st.columns(2)
                if ca.button("💾 Store Memory", use_container_width=True, type="primary"):
                    if canvas_train.image_data is not None and t_name.strip() != "":
                        bipolar = process_canvas(canvas_train.image_data)
                        if (bipolar == -1).all():
                            st.error("Canvas is blank!")
                        else:
                            st.session_state.hop_memories[t_name.strip()] = bipolar
                            net.train(list(st.session_state.hop_memories.values()))
                            st.session_state.ai_analysis = analyze_network_state_ai() # refresh
                            st.session_state.canvas_trn_key += 1
                            st.success(f"'{t_name}' mathematically locked into the synaptic weights.")
                            st.rerun()
                    elif t_name.strip() == "":
                        st.warning("Please name your diagram before storing.")
                        
                if cb.button("🗑️ Clear Training Board", use_container_width=True):
                    st.session_state.canvas_trn_key += 1
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color:white; margin-top:0;">📚 Currently Stored Diagrams</h3>', unsafe_allow_html=True)
            if len(st.session_state.hop_memories) == 0:
                st.markdown('<div style="text-align:center; padding: 50px; color:#A0AEC0;">The brain is completely empty. Draw a diagram on the left to begin training.</div>', unsafe_allow_html=True)
            else:
                mem_cols = st.columns(3)
                for i, (name, pat) in enumerate(st.session_state.hop_memories.items()):
                    with mem_cols[i % 3]:
                        st.plotly_chart(plot_matrix(pat, name), use_container_width=True)
                
                if st.button("💥 Reset Brain (Forget All)"):
                    st.session_state.hop_memories = {}
                    net.train([])
                    st.session_state.recovered = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # TAB 2: TESTING
    # -----------------------------------------------------------
    with t2:
        if len(st.session_state.hop_memories) == 0:
            st.warning("You must store at least ONE custom diagram in 'Step 1' before you can test the network!")
        else:
            c3, c4 = st.columns([1, 1.2])
            with c3:
                st.markdown('<div class="premium-card neon-pulse-border">', unsafe_allow_html=True)
                st.markdown('<h3 style="color:white; margin-top:0;">2. The Recall Tester</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color:#A0AEC0; font-size:0.9rem;">Draw a <b>corrupted or messy</b> version of your stored diagram. The network will mathematically rebuild it.</p>', unsafe_allow_html=True)
                
                if CANVAS_OK:
                    canvas_test = st_canvas(
                        fill_color="rgba(0,0,0,0)", stroke_width=25, stroke_color="#8B5CF6",
                        background_color="#0f172a", height=300, width=300, drawing_mode="freedraw",
                        key=f"hw_tst_{st.session_state.canvas_tst_key}"
                    )
                    
                    cc_a, cc_b = st.columns(2)
                    if cc_a.button("🧠 Execute Reconstruction", use_container_width=True, type="primary"):
                        if canvas_test.image_data is not None:
                            bipolar = process_canvas(canvas_test.image_data)
                            if not (bipolar == -1).all():
                                st.session_state.user_input = bipolar
                                st.session_state.recovered, st.session_state.energies = net.recover(bipolar, steps=150)
                                st.session_state.ai_analysis = analyze_network_state_ai()
                                
                    if cc_b.button("🗑️ Clear Testing Board", use_container_width=True):
                        st.session_state.user_input = None
                        st.session_state.recovered = None
                        st.session_state.canvas_tst_key += 1
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color:white; margin-top:0;">⚙️ Convergence Telemetry</h3>', unsafe_allow_html=True)
                if st.session_state.get("recovered") is not None:
                    out = st.session_state.recovered
                    overlaps = {k: float(out @ v) / N for k, v in st.session_state.hop_memories.items()}
                    best_match = max(overlaps, key=overlaps.get)
                    
                    st.markdown(f'<div style="text-align:center; padding: 10px;"><h2 style="color:#00ffcc; margin:0;">Matched: {best_match}</h2></div>', unsafe_allow_html=True)
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.plotly_chart(plot_matrix(st.session_state.user_input, "Noisy Input"), use_container_width=True)
                    with rc2:
                        st.plotly_chart(plot_matrix(out, "Flawless Reconstruction"), use_container_width=True)
                else:
                    st.markdown('<div style="text-align:center; padding: 40px; color:#A0AEC0;">Waiting for sensory input...</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # ARCHITECTURE & LOGIC (Expandable Breakdown)
    # -----------------------------------------------------------
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    with st.expander("🛠️ Internal Working Steps, Logic Architecture & Outputs"):
        st.markdown('<div class="premium-card" style="margin-top: 10px;">', unsafe_allow_html=True)
        st.markdown("## 1. Network Topology Analysis")
        st.markdown("The Network uses the **Moore-Penrose Pseudoinverse (Projection) Learning Rule**. This means the synaptic weight matrix $W$ relies on $W = X \cdot X^+$.")
        analyst_text = st.session_state.get("ai_analysis", analyze_network_state_ai())
        st.markdown(f'<div style="background:rgba(59, 130, 246, 0.1); border-left:4px solid #3B82F6; padding:15px; border-radius:4px; color:#E2E8F0; font-family:\'JetBrains Mono\';">🤖 <b>AI Network Analyst:</b><br>{analyst_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 2. Live Synaptic Weight Matrix ($W$)")
        st.markdown("This matrix defines the energy landscape. The network mathematically rolls down the topology dictated by these weights.")
        st.plotly_chart(plot_weight_matrix(net.W), use_container_width=True)
        
        st.markdown("---")
        if st.session_state.get("recovered") is not None:
            st.markdown("## 3. Mathematical Descent (Energy Step Tracking)")
            st.markdown("The network updates asyncronously. With every step, the global energy strictly decreases until it reaches an absolute local minimum (the attractor).")
            st.plotly_chart(plot_energy(st.session_state.energies), use_container_width=True)
            
            st.markdown("## 4. Final Reconstructed Output Tensor")
            st.markdown("The complete output vector array after recovery:")
            out_str = np.array2string(st.session_state.recovered, max_line_width=80, separator=', ')
            st.code(out_str, language="python")
        else:
            st.markdown("## 3/4. Mathematical Descent & Output Tensor")
            st.info("Execute a reconstruction in Step 2 to generate the mathematical energy traces and output tensors.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()