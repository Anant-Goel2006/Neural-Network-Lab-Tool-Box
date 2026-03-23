import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
from utils.styles import gradient_header, section_header, render_log, render_nlp_insight
from utils.nlp_engine import generate_bwd_insight
from utils.nn_helpers import (ACTS, LOSSES, make_weights, forward_pass, backward_pass,
    draw_network, P, C, G, A, R, TEXT, MUTED, GRID, PLOTLY_BASE, plotly_layout)

def _wkey(n, h): return f"bp_w_{n}_{'_'.join(str(x) for x in h)}"
def _get_w(n, h):
    k = _wkey(n, h)
    if k not in st.session_state: st.session_state[k] = make_weights(n, h)
    return st.session_state[k]

def backward_propagation_page():
    from utils.styles import inject_global_css
    inject_global_css()
    gradient_header("Backward Propagation",
        "Live Chain Rule & Gradient Flow Training · Watch Weights Update in Real-Time", "⬅️")

    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown("""
        **The Backward Pass (Backpropagation):**
        1. **Compute Network Error:** Find the discrepancy between `ŷ` and `y`.
        2. **Gradients via Chain Rule:** Propagate the error backwards:
           `dL/dW = dL/dA × dA/dZ × dZ/dW`
           - This tracks exactly how much each weight contributed to the total error.
        3. **Vanishing Gradients:** If `dA/dZ` (activation derivative) is too small, gradients "vanish" making early layers learn too slowly.
        4. **Weight Update:** `W_new = W - η·dL/dW`
        """)

    st.divider()
    section_header("1. Network Architecture", "Define MLP topology for real-time training")
    c1, c2 = st.columns(2)
    n_in = c1.slider("Input features", 1, 10, 2)
    n_hid = c2.slider("Hidden layers", 1, 4, 1)
    hid_sz = []
    nc = st.columns(min(n_hid, 4))
    for l in range(n_hid):
        hid_sz.append(nc[l].slider(f"H{l+1} Width", 1, 10, 3, key=f"bp_hl_{l}"))
    sizes = [n_in] + hid_sz + [1]
    labels = ["Input"] + [f"H{i+1}" for i in range(n_hid)] + ["Output"]
    st.markdown(" → ".join([f"**{lb}**({sz})" for lb, sz in zip(labels, sizes)]))

    st.divider()
    section_header("2. Inputs & Target", "Set the training sample")
    ic = st.columns(min(n_in+1, 5)); Xv = []
    for i in range(n_in):
        Xv.append(ic[i%4].number_input(f"x{i+1}", value=round(np.random.rand(), 2), step=0.1, key=f"bp_x{i}"))
    y_true = ic[min(n_in, 4)].number_input("Target y", value=1.0, step=0.1, key="bp_yt")
    X = np.array(Xv).reshape(-1, 1)

    st.divider()
    section_header("3. Hyperparameters & Activations", "Configure training rules")
    h1, h2, h3 = st.columns(3)
    lr = h1.number_input("Learning rate η", 0.001, 2.0, 0.1, 0.01)
    max_ep = h2.slider("Training Epochs", 10, 500, 100)
    delay = h3.slider("Animation Delay (s)", 0.0, 0.5, 0.05, 0.05)

    a1, a2, a3 = st.columns(3)
    act_all = a1.selectbox("Hidden Activation", list(ACTS.keys()), index=0)
    h_acts = [act_all] * n_hid
    o_act = a2.selectbox("Output Activation", list(ACTS.keys()), index=1)
    loss_fn = a3.selectbox("Loss Function", list(LOSSES.keys()), index=1)

    weights = make_weights(n_in, hid_sz) # Fresh random init

    st.divider()
    if st.button("🚀 Start Live Backprop Training", type="primary", width="stretch"):
        master_dashboard = st.empty()
        
        losses = []
        y_preds = []
        grad_mags = []

        for ep in range(1, max_ep+1):
            # Forward Pass
            Zs, As = forward_pass(X, weights, h_acts, o_act)
            y_pred = float(As[-1][0][0])
            loss = float(LOSSES[loss_fn]["fn"](As[-1], np.array([[y_true]])))
            
            # Backward Pass
            grads = backward_pass(weights, As, y_true, h_acts, o_act, loss_fn)
            
            # Gradient logging
            mean_grad = np.mean([np.mean(np.abs(g["dLdW"])) for g in grads if g])
            
            # Weight Update
            new_weights = []
            for (W, b_w), g in zip(weights, grads):
                W_new = W - lr * g["dLdW"]
                b_new = b_w - lr * g["dLdb"]
                new_weights.append((W_new, b_new))
            weights = new_weights
            
            losses.append(loss)
            y_preds.append(y_pred)
            grad_mags.append(mean_grad)
            
            with master_dashboard.container():
                mx1, mx2, mx3, mx4 = st.columns(4)
                mx1.metric("Epoch", f"{ep}/{max_ep}"); mx2.metric("Loss", f"{loss:.6f}")
                # Calculate a simple 'accuracy' for display purposes, e.g., inverse of loss or proximity to target
                # This is a placeholder and might need a more robust definition depending on the loss function
                # For MSE, a simple inverse or 1 - normalized_loss could be used.
                # For demonstration, let's use a simple proximity measure.
                # If y_true is 1.0, and y_pred is 0.9, error is 0.1. Accuracy could be 1 - 0.1 = 0.9.
                # Max possible error for a single output is usually bounded (e.g., 1.0 for binary classification with BCE, or larger for MSE).
                # Let's assume a max possible loss of 1.0 for a rough 'accuracy' gauge.
                # This is a heuristic for visualization, not a true accuracy metric.
                max_display_loss = 1.0 # Adjust based on expected loss range
                display_acc = max(0, 1 - (loss / max_display_loss)) # Simple inverse of normalized loss
                mx3.metric("Prediction ŷ", f"{y_pred:.6f}"); mx4.metric("Avg Grad", f"{mean_grad:.6f}")
                
                # Restore Gauges
                gA, gB = st.columns(2)
                from utils.styles import speedometer
                gA.plotly_chart(speedometer(loss, max_display_loss, "Current Loss", color="#EF4444", height=200), width="stretch", theme=None, key=f"bp_gL_{ep}")
                gB.plotly_chart(speedometer(mean_grad, 1.0, "Avg Gradient", color="#3B82F6", height=200), width="stretch", theme=None, key=f"bp_gA_{ep}")
                
                # Double graph: Network Architecture (Live Vals) + Trajectories
                diag_vals = [Xv] + [As[i+1].flatten().tolist() for i in range(len(sizes)-2)] + [[y_pred]]
                fig1 = draw_network(sizes, labels, vals=diag_vals)
                fig1.update_layout(title=dict(text="Live Network Activations"), **plotly_layout(height=350, margin=dict(l=0,r=0,t=40,b=0)))
                
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=list(range(1, ep+1)), y=losses, mode="lines", name="Loss", line=dict(color="#EF4444", width=3)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=list(range(1, ep+1)), y=grad_mags, mode="lines", name="Avg Gradient", line=dict(color="#FACC15", width=2, dash="dash")), secondary_y=True)
                fig2.add_hline(y=0, line=dict(color="#333", width=1))
                fig2.update_layout(title_text=f"Loss & Gradient Flow (Epoch {ep})", 
                                   **plotly_layout(height=350, margin=dict(l=40,r=40,t=40,b=40),
                                                   title=dict(font=dict(family="Montserrat", size=24))))
                fig2.update_xaxes(showgrid=True, gridcolor="#DDD", zerolinecolor="#555")
                fig2.update_yaxes(showgrid=True, gridcolor="#DDD", secondary_y=False)
                
                cA, cB = st.columns(2)
                cA.plotly_chart(fig1, width="stretch", theme=None, key=f"bp_net_live_{ep}")
                cB.plotly_chart(fig2, width="stretch", theme=None, key=f"bp_line_live_{ep}")

            if delay > 0: time.sleep(delay)
            if loss < 1e-5:
                break
                
        # Calculate final accuracy for the display card
        final_max_display_loss = 1.0 # Re-define as it was local to the loop
        acc = max(0, 1 - (loss / final_max_display_loss))

        with master_dashboard.container():
            insight = generate_bwd_insight(optimizer="Gradient Descent Kinetics", lr=lr, total_epochs=ep)
            render_nlp_insight(insight, "Gradient Descent Log // NLP Neural Engine", "#FACC15")
            
            st.divider()
            section_header("Verify Result", "Final Prediction Quality")
            st.markdown(f"""
            <div class="premium-card fade-in" style="text-align:center; border-top: 5px solid #3B82F6; padding:60px 40px;">
                <div style="font-family:'Montserrat', sans-serif; font-size:16px; color:#94A3B8; letter-spacing:3px; font-weight: 700; text-transform: uppercase;">Neural Convergence Scan</div>
                <div style="font-size:72px; font-family:'Montserrat', sans-serif; font-weight: 800; color:#FFFFFF; margin: 30px 0; line-height:1; text-shadow: 0 0 30px rgba(59, 130, 246, 0.4);">
                    {acc*100:.2f}%
                </div>
                <div style="font-weight:700; font-family:'Inter', sans-serif; background:rgba(255,255,255,0.05); display:inline-block; padding:12px 24px; border-radius: 50px; border:1px solid rgba(255,255,255,0.1); color:#FACC15; text-transform:uppercase; letter-spacing: 2px; font-size:18px;">
                    Stability Rating: <span style="color:#22C55E;">{'STABILIZED' if acc > 0.95 else 'ADAPTIVE'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; gap:20px; margin: 40px 0; flex-wrap:wrap;">
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #EF4444;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Training Epoch</div>
                    <div style="color:#FFFFFF; font-size:32px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{ep}/{max_ep}</div>
                </div>
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #3B82F6;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Residual Error</div>
                    <div style="color:#FFFFFF; font-size:32px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{loss:.5f}</div>
                </div>
                <div class="premium-card fade-in" style="flex:1; min-width:180px; padding:20px; text-align:center; border-top: 3px solid #22C55E;">
                    <div style="color:#94A3B8; font-size:12px; font-weight:700; font-family:'Inter', sans-serif; letter-spacing:2px; text-transform: uppercase;">Model Pred ŷ</div>
                    <div style="color:#FFFFFF; font-size:32px; font-weight:800; font-family:'Montserrat', sans-serif; margin-top:10px;">{y_pred:.5f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            cA, cB = st.columns(2)
            cA.plotly_chart(fig1, width="stretch", theme=None, key="bp_net_final")
            cB.plotly_chart(fig2, width="stretch", theme=None, key="bp_line_final")

        # ── RESTORED POST-TRAINING MAPS ──
        st.divider()
        section_header("Post-Training Analytics", "Detailed inspection of intermediate gradients & activations")
        
        t1, t2 = st.tabs(["📉 Gradient Magnitudes", "⚖ Final Weights Array"])
        
        with t1:
            df_g = pd.DataFrame({"Layer": labels[1:], "Avg |dL/dW|": grad_mags[-abs(n_hid)-1:] if len(grad_mags)>n_hid else grad_mags})
            st.dataframe(df_g, hide_index=True, width="stretch")
            st.info("If gradients fall below 0.1, the network may suffer from the Vanishing Gradient problem.")
        
        with t2:
            st.write("### Model Final Weights")
            for i, (W, b) in enumerate(weights):
                lbl = "Output Layer" if i == len(weights)-1 else f"Hidden {i+1} Weights"
                df_w = pd.DataFrame(W)
                df_w["Bias"] = b.flatten()
                st.caption(f"**{lbl}**")
                st.dataframe(df_w, width="stretch")
