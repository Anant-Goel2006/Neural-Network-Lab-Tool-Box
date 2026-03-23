import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
from utils.styles import section_header, render_log, render_nlp_insight, gradient_header
from utils.nlp_engine import generate_perceptron_insight
from utils.nn_helpers import P, C, G, A, R, TEXT, MUTED, GRID, PLOTLY_BASE, plotly_layout

GATES = {
    "AND":  {"data":[(0,0,0),(0,1,0),(1,0,0),(1,1,1)],"sep":True, "icon":"⊗"},
    "OR":   {"data":[(0,0,0),(0,1,1),(1,0,1),(1,1,1)],"sep":True, "icon":"⊕"},
    "NAND": {"data":[(0,0,1),(0,1,1),(1,0,1),(1,1,0)],"sep":True, "icon":"↑"},
    "NOR":  {"data":[(0,0,1),(0,1,0),(1,0,0),(1,1,0)],"sep":True, "icon":"↓"},
    "XOR":  {"data":[(0,0,0),(0,1,1),(1,0,1),(1,1,0)],"sep":False,"icon":"⊻"},
    "XNOR": {"data":[(0,0,1),(0,1,0),(1,0,0),(1,1,1)],"sep":False,"icon":"⊙"},
}

def _live_dashboard_fig(X, y, w, b, losses, acc, ep, max_ep):
    # 1x2 Subplots: [Decision Boundary] | [Loss Curve]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Decision Boundary", "Live Loss Curve"))
    
    # Boundary
    for cls, col, sym in [(0, R, "circle"), (1, G, "diamond")]:
        m = y == cls
        if m.any():
            fig.add_trace(go.Scatter(x=X[m,0], y=X[m,1], mode="markers",
                name=f"Class {cls}",
                marker=dict(size=14, color=col, line=dict(width=2, color="#000"), symbol=sym)),
                row=1, col=1)
    
    if abs(w[1]) > 1e-9:
        xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300)
        ys = -(w[0]*xs+b)/w[1]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Boundary",
            line=dict(color="#A78BFA", width=3, dash="dash")), row=1, col=1)
    
    # Loss Curve
    ep_x = list(range(1, len(losses)+1))
    fig.add_trace(go.Scatter(x=ep_x, y=losses, mode="lines", fill="tozeroy",
        name="Total Error", line=dict(color="#06B6D4", width=3),
        fillcolor="rgba(6,182,212,0.15)"), row=1, col=2)
    
    fig.update_layout(
        title_text=f"Live Convergence Dashboard — Epoch {ep}/{max_ep} | Accuracy: {acc:.1f}%",
        showlegend=True,
        **plotly_layout(height=420, margin=dict(l=40, r=40, t=60, b=40))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333", zerolinecolor="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#333", zerolinecolor="#555")
    return fig

def perceptron_page():
    from utils.styles import inject_global_css
    inject_global_css()
    gradient_header("The Perceptron", "Binary Linear Classifier · Stable Live Visualization", "🧠")
    
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown("""
        **The Perceptron Learning Rule:**
        1. Calculate the weighted sum of inputs: `z = w·x + b`
        2. Apply the heavy-side step activation to get prediction `ŷ`: `ŷ = 1 if z >= 0 else 0`
        3. Compare with actual target `y` and compute error: `e = y - ŷ`
        4. Update weights using learning rate `η`:
           `w_new = w + η·e·x` | `b_new = b + η·e`
        """)

    # Init
    if "pc_w" not in st.session_state:
        st.session_state.pc_tied = False

    st.divider()
    section_header("1. Dataset Selection", "Choose linearly separable constraints")
    with st.container(border=True):
        c1, c2 = st.columns([1,1])
        with c1:
            gate = st.selectbox("Logic Gate", list(GATES.keys()), index=0)
            ginfo = GATES[gate]
            raw = ginfo["data"]
            X = np.array([[r[0],r[1]] for r in raw], dtype=float)
            y = np.array([r[2] for r in raw], dtype=int)
            if not ginfo["sep"]: st.warning("Not linearly separable (Will never reach 0 loss)")
        with c2:
            df_tt = pd.DataFrame(raw, columns=["X1","X2","Outputs"])
            st.dataframe(df_tt, use_container_width=True, hide_index=True)

    st.divider()
    section_header("2. Hyperparameters", "Tuning controls")
    with st.container(border=True):
        h1, h2, h3 = st.columns(3)
        lr = h1.number_input("Learning Rate (η)", 0.001, 1.0, 0.1, 0.01)
        max_ep = h2.slider("Max Epochs", 10, 500, 100)
        delay = h3.slider("Animation Delay (s)", 0.0, 0.5, 0.05, 0.05)

    if st.button("🚀 Start Live Training", type="primary", width="stretch"):
        master_dashboard = st.empty()
        
        w = np.random.uniform(-0.5, 0.5, 2)
        b = np.random.uniform(-0.5, 0.5)
        losses = []
        w_traj = []
        
        for ep in range(1, max_ep+1):
            err = 0
            for xi, yi in zip(X, y):
                z = np.dot(w, xi) + b
                pred = 1 if z >= 0 else 0
                e = yi - pred
                err += abs(e)
                if e != 0:
                    w += lr * e * xi
                    b += lr * e
            
            losses.append(err)
            w_traj.append([w[0], w[1], b])
            corr = int(np.sum((X@w+b>=0).astype(int)==y))
            acc = corr / len(y) * 100
            # --- Rendering strictly inside one container to absolutely prevent native overlapping ---
            with master_dashboard.container():
                mx1, mx2, mx3, mx4 = st.columns(4)
                mx1.metric("Epoch", f"{ep}/{max_ep}")
                mx2.metric("Loss", f"{err:.1f}")
                mx3.metric("Accuracy", f"{acc:.1f}%")
                mx4.metric("Weights", f"w1:{w[0]:.2f} w2:{w[1]:.2f} b:{b:.2f}")
                
                from utils.styles import speedometer
                st.plotly_chart(speedometer(acc, 100, "Accuracy %", color="#005BEA", height=220), width="stretch", theme=None, key=f"pct_gauge_{ep}")
                
                fig = _live_dashboard_fig(X, y, w, b, losses, acc, ep, max_ep)
                st.plotly_chart(fig, width="stretch", theme=None, key=f"pct_live_{ep}")
            
            if delay > 0: time.sleep(delay)
            if err == 0:
                break
                
        # --- MANUAL PREDICTOR ---
        st.divider()
        section_header("Verify Result", "Test custom inputs on your trained model")
        with st.expander("🕵️ Open Manual Predictor", expanded=True):
            c1, c2 = st.columns(2)
            ix1 = c1.number_input("Input X1", value=0.0)
            ix2 = c2.number_input("Input X2", value=1.0)
            
            z = w[0]*ix1 + w[1]*ix2 + b
            y_pred_m = 1 if z >= 0 else 0
            
            st.markdown(f"""
            <div class="premium-card fade-in" style="text-align:center; border-top: 4px solid {G if y_pred_m==1 else R};">
                <div style="font-family:'Montserrat', sans-serif; font-size:16px; color:#94A3B8; letter-spacing:2px; text-transform: uppercase; font-weight: 600;">Neural Prediction Result</div>
                <div style="font-size:54px; color:{G if y_pred_m==1 else R}; font-family:'Montserrat', sans-serif; font-weight: 800; margin: 20px 0; text-shadow: 0 0 20px rgba({int((G if y_pred_m==1 else R)[1:3], 16)}, {int((G if y_pred_m==1 else R)[3:5], 16)}, {int((G if y_pred_m==1 else R)[5:7], 16)}, 0.3);">
                    {'POSITIVE (+1)' if y_pred_m==1 else 'NEGATIVE (0)'}
                </div>
                <div style="font-size: 13px; color: #64748B; font-weight: 500;">Activation: {'Step (Heaviside)'}</div>
            </div>
            """, unsafe_allow_html=True)

        with master_dashboard.container():
            st.success(f"Training finalized at Epoch {ep} with Accuracy {acc:.1f}%.")
            gradient_header("Perceptron Processor", "Basic Linear Classification", "📉")
            reg_name = st.sidebar.text_input("Analysis Target Name", "Perceptron_Alpha", key="p_name")
            insight = generate_perceptron_insight(ep, acc/100, err, acc == 100.0)
            render_nlp_insight(insight, "Synaptic Mission Intel // Analyzing Patterns", "#FACC15")
            
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; gap:20px; margin-bottom: 40px; flex-wrap:wrap;">
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #60A5FA;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Current Epoch</div>
                    <div style="color:#FFFFFF; font-size:38px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{ep}</div>
                </div>
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #EF4444;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Signal Loss</div>
                    <div style="color:#FFFFFF; font-size:38px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{err:.2f}</div>
                </div>
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #10B981;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Accuracy</div>
                    <div style="color:#FFFFFF; font-size:38px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{acc:.1f}%</div>
                </div>
                <div class="premium-card fade-in" style="flex:2; min-width:240px; padding:20px; text-align:center; border-top: 3px solid #FACC15;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Final Weights</div>
                    <div style="color:#FFFFFF; font-size:16px; font-weight:600; font-family:'JetBrains Mono', monospace; margin-top:16px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px;">
                        w1:{w[0]:.2f} | w2:{w[1]:.2f} | b:{b:.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = _live_dashboard_fig(X, y, w, b, losses, acc, ep, max_ep)
            st.plotly_chart(fig, width="stretch", theme=None, key="pct_final_res")

        # ── RESTORED POST-TRAINING ANALYTICS ──
        st.divider()
        section_header("Post-Training Analytics", "Detailed inspection of the trained perceptron")
        
        t1, t2, t3 = st.tabs(["🔢 Confusion Matrix", "📈 Weight Trajectory", "📊 Distributions"])
        
        with t1:
            preds = (X @ w + b >= 0).astype(int)
            tp = int(np.sum((preds==1)&(y==1))); fp = int(np.sum((preds==1)&(y==0)))
            fn = int(np.sum((preds==0)&(y==1))); tn = int(np.sum((preds==0)&(y==0)))
            cmat = pd.DataFrame([[tp, fp], [fn, tn]], columns=["Pred 1", "Pred 0"], index=["Actual 1", "Actual 0"])
            st.dataframe(cmat, width="stretch")
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0; rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            st.info(f"**Precision:** {prec:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; **Recall:** {rec:.2f}")
            
        with t2:
            df_w = pd.DataFrame(w_traj, columns=["w1", "w2", "bias"])
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(y=df_w["w1"], name="w1"))
            fig_w.add_trace(go.Scatter(y=df_w["w2"], name="w2"))
            fig_w.add_trace(go.Scatter(y=df_w["bias"], name="bias", line=dict(dash="dot")))
            fig_w.update_layout(title="Weights across Epochs", height=300, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_w, width="stretch", theme=None, key="pct_weights_traj")
            
        with t3:
            z = X @ w + b
            df_dist = pd.DataFrame({"z_score": z, "Actual Class": y})
            st.dataframe(df_dist, width="stretch", hide_index=True)
