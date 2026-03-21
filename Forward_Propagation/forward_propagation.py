import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.styles import gradient_header, section_header, render_log, render_nlp_insight
from utils.nlp_engine import generate_fwd_insight
from utils.nn_helpers import (ACTS, LOSSES, make_weights, forward_pass,
    draw_network, P, C, G, A, R, TEXT, MUTED, GRID, PLOTLY_BASE, LAYER_COLS, plotly_layout)

MAX_NODES = 6; MAX_LAYERS = 7

def _wkey(n, h): return f"fp_w_{n}_{'_'.join(str(x) for x in h)}"
def _get_w(n, h):
    k = _wkey(n, h)
    if k not in st.session_state: st.session_state[k] = make_weights(n, h)
    return st.session_state[k]

def _act_curve_fig(act_name):
    xs = np.linspace(-6, 6, 300); fn = ACTS[act_name]["fn"]; ys = fn(xs)
    xs2 = np.linspace(-6, 6, 300) + 0.001; deriv = (fn(xs2) - fn(xs)) / 0.001
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
        name=f"f(x) — {ACTS[act_name]['eq']}",
        line=dict(color="#8B5CF6", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=np.clip(deriv, -5, 5), mode="lines",
        name="f'(x)", line=dict(color="#06B6D4", width=2, dash="dot")))
    fig.add_hline(y=0, line=dict(color=MUTED, width=0.5))
    fig.add_vline(x=0, line=dict(color=MUTED, width=0.5))
    fig.update_layout(
        title=dict(text=f"{act_name} Activation", font=dict(color=TEXT, family="Impact", size=16)),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        **plotly_layout(
            xaxis=dict(gridcolor=GRID, color=MUTED, range=[-6,6]),
            yaxis=dict(gridcolor=GRID, color=MUTED, range=[-1.5,1.5]),
            height=220
        ))
    return fig

def _activation_heatmap(As, labels):
    rows = []; lbs = []
    for i, act_vals in enumerate(As):
        flat = act_vals.flatten()[:8]
        if len(flat) > 0:
            rows.append(list(flat) + [np.nan]*(8-len(flat)))
            lbs.append(labels[i] if i < len(labels) else f"L{i}")
    if not rows: return None
    fig = go.Figure(go.Heatmap(z=rows, x=[f"n{i+1}" for i in range(8)], y=lbs,
        colorscale=[[0,"#ED1D24"],[0.5,"#FFFFFF"],[1,"#005BEA"]],
        zmid=0, colorbar=dict(title="Activation", tickfont=dict(color=TEXT)),
        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in rows],
        texttemplate="%{text}", textfont=dict(size=10)))
    fig.update_layout(
        title=dict(text="Activation Heatmap", font=dict(color=TEXT, family="Impact", size=16)),
        **plotly_layout(
            xaxis=dict(color=MUTED), 
            yaxis=dict(color=MUTED), 
            height=240
        ))
    return fig

def forward_propagation_page():
    from utils.styles import inject_global_css
    inject_global_css()
    gradient_header("Forward Propagation",
        "Layer-by-Layer Signal Flow · Z = W·A + b → A = activation(Z) · Live Heatmaps", "➡️")

    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown("""
        **The Forward Pass:**
        The multi-layer perceptron (MLP) processes data sequentially:
        1. **Pre-activation:** `Z = W·A_prev + b`
           - Computes the linear transformation of previous activations.
        2. **Activation:** `A = f(Z)`
           - Applies a non-linear squashing function (e.g. ReLU, Sigmoid).
        3. **Output:** The final layer produces `ŷ`, which is compared against the target `y` using a Loss Function.
        """)

    st.divider()
    section_header("Network Architecture Builder", "Define your neural network topology")
    c1, c2, c3 = st.columns(3)
    n_in  = c1.slider("Input features", 1, 20, 3)
    n_hid = c2.slider("Hidden layers", 1, 6, 2)
    same  = c3.checkbox("Uniform hidden width", True)

    hid_sz = []
    if same:
        w = st.slider("Neurons per hidden layer", 1, 20, 4)
        hid_sz = [w] * n_hid
    else:
        nc = st.columns(min(n_hid, 5))
        for l in range(n_hid):
            hid_sz.append(nc[l%5].slider(f"H{l+1}", 1, 20, 3, key=f"fp_hl_{l}"))

    sizes = [n_in] + hid_sz + [1]
    labels = ["Input"] + [f"H{i+1}" for i in range(n_hid)] + ["Output"]

    # Architecture badge
    arch = " → ".join([f"**{lb}**({sz})" for lb, sz in zip(labels, sizes)])
    st.markdown(arch)

    if len(sizes) <= MAX_LAYERS:
        st.plotly_chart(draw_network(sizes, labels), use_container_width=True, theme=None)

    st.divider()
    section_header("Inputs & Target", "Set input values and target output")
    ic = st.columns(min(n_in+1, 5)); Xv = []
    for i in range(n_in):
        Xv.append(ic[i%4].number_input(f"x{i+1}", value=round(0.3+i*0.15, 2), step=0.1, key=f"fp_x{i}"))
    y_true = ic[min(n_in, 4)].number_input("Target y", value=1.0, step=0.1, key="fp_yt")
    X = np.array(Xv).reshape(-1, 1)

    st.divider()
    section_header("Activation Functions", "Configure per-layer activations and loss function")
    a1, a2, a3, a4 = st.columns(4)
    same_act = a1.checkbox("Same for all hidden", True)
    if same_act:
        act_all = a2.selectbox("Hidden activation", list(ACTS.keys()))
        st.caption(f"f(z) = `{ACTS[act_all]['eq']}`")
        h_acts = [act_all] * n_hid
        with st.expander("📈 Activation curve"):
            st.plotly_chart(_act_curve_fig(act_all), use_container_width=True, theme=None)
    else:
        h_acts = []
        ac = st.columns(min(n_hid, 5))
        for l in range(n_hid):
            h_acts.append(ac[l%5].selectbox(f"H{l+1}", list(ACTS.keys()), key=f"fp_a_{l}"))
    o_act   = a3.selectbox("Output activation", list(ACTS.keys()), index=1)
    loss_fn = a4.selectbox("Loss function", list(LOSSES.keys()))
    st.caption(f"Loss: `{LOSSES[loss_fn]['eq']}`")

    st.divider()
    section_header("Weight Management", "Randomize or manually set all weights and biases")
    can_man = n_in <= 6 and all(h <= 6 for h in hid_sz)
    mode = st.radio("Mode", ["Random","Manual"], horizontal=True) if can_man else "Random"
    if not can_man: st.warning("Manual entry disabled for large networks.")
    weights = _get_w(n_in, hid_sz)

    if mode == "Random":
        if st.button("🎲 Randomize Weights"):
            st.session_state[_wkey(n_in, hid_sz)] = make_weights(n_in, hid_sz); st.rerun()
        weights = _get_w(n_in, hid_sz)
        with st.expander("View Weights", expanded=False):
            for li, (W, b_w) in enumerate(weights):
                lbl = "Output" if li == len(weights)-1 else f"Hidden {li+1}"
                df = pd.DataFrame(W, columns=[f"in{i+1}" for i in range(W.shape[1])],
                    index=[f"n{j+1}" for j in range(W.shape[0])])
                df["bias"] = b_w.flatten()
                st.caption(f"**{lbl}** W{W.shape}")
                st.dataframe(df.round(4), use_container_width=True)
    else:
        wm = []; sz = n_in
        for li, h in enumerate(hid_sz):
            Wm = np.zeros((h, sz)); bm = np.zeros((h, 1))
            with st.expander(f"Hidden {li+1} W({h}×{sz})", expanded=li==0):
                for j in range(h):
                    row = st.columns(sz+1)
                    for i in range(sz):
                        Wm[j,i] = row[i].number_input(f"W[{j+1},{i+1}]",
                            value=0.5 if i==j else 0.0, min_value=-2.0, max_value=2.0,
                            step=0.1, key=f"fp_mw_{li}_{j}_{i}")
                    bm[j,0] = row[sz].number_input(f"b[{j+1}]",
                        value=0.0, min_value=-2.0, max_value=2.0,
                        step=0.1, key=f"fp_mb_{li}_{j}")
            wm.append((Wm, bm)); sz = h
        Wo = np.zeros((1,sz)); bo = np.zeros((1,1))
        with st.expander(f"Output W(1×{sz})", expanded=True):
            oc = st.columns(sz+1)
            for j in range(sz):
                Wo[0,j] = oc[j].number_input(f"Wo[{j+1}]", value=1.0,
                    min_value=-2.0, max_value=2.0, step=0.1, key=f"fp_mwo_{j}")
            bo[0,0] = oc[sz].number_input("bo", value=0.0, min_value=-2.0,
                max_value=2.0, step=0.1, key="fp_mbo")
        wm.append((Wo, bo)); weights = wm

    st.divider()
    bc, rc = st.columns([4,1])
    with rc:
        if st.button("Reset", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("fp_")]: del st.session_state[k]
            st.rerun()
    log_ph = st.expander("📋 Computation Log", expanded=False).empty()

    with bc:
        run_btn = st.button("▶ Run Forward Pass", type="primary", use_container_width=True)

    if run_btn:
        Zs, As = forward_pass(X, weights, h_acts, o_act)
        y_pred = float(As[-1][0][0])
        loss = float(LOSSES[loss_fn]["fn"](As[-1], np.array([[y_true]])))
        log = ["FORWARD PROPAGATION\n", "═"*60+"\n",
               f"Inputs: {[f'x{i+1}={v:.4f}' for i,v in enumerate(Xv)]}\n\n"]
        for li, (W, b_w) in enumerate(weights):
            Ap = As[li]; Z = Zs[li]; Act = As[li+1]
            is_o = li == len(weights)-1
            act = o_act if is_o else h_acts[li]
            lbl = "OUTPUT" if is_o else f"HIDDEN {li+1}"
            log.append(f"{lbl} [{act}]  Z=W·A+b  →  A=activation(Z)\n")
            if W.shape[0] <= 6 and W.shape[1] <= 6:
                for j in range(W.shape[0]):
                    terms = " + ".join([f"({W[j,i]:.3f}×{Ap[i,0]:.3f})" for i in range(W.shape[1])])
                    log.append(f"  n{j+1}: {terms} + {b_w[j,0]:.3f} = {Z[j,0]:.4f} → {Act[j,0]:.4f}\n")
            else:
                log.append(f"  Z[:4]={np.round(Z.flatten()[:4],4).tolist()}...\n")
                log.append(f"  A[:4]={np.round(Act.flatten()[:4],4).tolist()}...\n")
            log.append("\n")
        log.append("═"*60+f"\nŷ = {y_pred:.6f}   y = {y_true:.4f}   Loss({loss_fn}) = {loss:.6f}\n")
        render_log(log_ph, log)
        st.session_state.update(dict(fp_Zs=Zs, fp_As=As, fp_loss=loss, fp_y_pred=y_pred,
            fp_sizes=sizes, fp_labels=labels, fp_Xv=Xv, fp_n_in=n_in, fp_hid=hid_sz,
            fp_y_true=y_true, fp_loss_fn=loss_fn, fp_h_acts=h_acts, fp_done=True))

    if not st.session_state.get("fp_done", False): return

    Zs = st.session_state.fp_Zs; As = st.session_state.fp_As
    loss = st.session_state.fp_loss; y_pred = st.session_state.fp_y_pred
    s_sizes = st.session_state.fp_sizes; s_lbl = st.session_state.fp_labels

    st.divider()
    section_header("Results Dashboard", "Output, loss gauge, network diagram, and layer-by-layer activations")

    insight = generate_fwd_insight(st.session_state.fp_h_acts[-1] if st.session_state.fp_h_acts else "Linear", st.session_state.fp_loss_fn, loss)
    render_nlp_insight(insight, "Feed-Forward Analysis // NLP Neural Engine", "#0ea5e9")

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; gap:16px; margin-bottom: 20px; flex-wrap:wrap;">
        <div style="flex:1; min-width:150px; background:rgba(10,10,20,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:16px; text-align:center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); word-wrap:break-word;">
            <div style="color:#a78bfa; font-size:12px; font-weight:400; font-family:'Inter'; text-transform:uppercase; letter-spacing:1px;">Output ŷ</div>
            <div style="color:#FAFAFA; font-size:24px; font-weight:700; font-family:'Oswald';">{y_pred:.6f}</div>
        </div>
        <div style="flex:1; min-width:150px; background:rgba(10,10,20,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:16px; text-align:center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); word-wrap:break-word;">
            <div style="color:#a78bfa; font-size:12px; font-weight:400; font-family:'Inter'; text-transform:uppercase; letter-spacing:1px;">Target y</div>
            <div style="color:#FAFAFA; font-size:24px; font-weight:700; font-family:'Oswald';">{st.session_state.fp_y_true:.4f}</div>
        </div>
        <div style="flex:1; min-width:150px; background:rgba(10,10,20,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:16px; text-align:center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); word-wrap:break-word;">
            <div style="color:#00f0ff; font-size:12px; font-weight:400; font-family:'Inter'; text-transform:uppercase; letter-spacing:1px;">{st.session_state.fp_loss_fn}</div>
            <div style="color:#FAFAFA; font-size:24px; font-weight:700; font-family:'Oswald'; filter:drop-shadow(0 0 5px rgba(0,240,255,0.4));">{loss:.6f}</div>
        </div>
        <div style="flex:1; min-width:150px; background:rgba(10,10,20,0.7); backdrop-filter:blur(12px); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:16px; text-align:center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); word-wrap:break-word;">
            <div style="color:#00f0ff; font-size:12px; font-weight:400; font-family:'Inter'; text-transform:uppercase; letter-spacing:1px;">Error</div>
            <div style="color:#00f0ff; font-size:24px; font-weight:700; font-family:'Oswald'; filter:drop-shadow(0 0 10px rgba(0,240,255,0.6));">{y_pred - st.session_state.fp_y_true:+.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauges restored
    g1, g2 = st.columns(2)
    from utils.styles import speedometer
    g1.plotly_chart(speedometer(y_pred, 2.0, "Prediction ŷ", color="#005BEA", height=250), use_container_width=True, theme=None, key="fp_g1")
    g2.plotly_chart(speedometer(loss, 1.0, "Loss Score", color="#ED1D24", height=250), use_container_width=True, theme=None, key="fp_g2")

    # Gauges removed for cleaner layout as requested.

    st.divider()
    section_header("Verify Result", "Compare Prediction vs Ground Truth")
    
    diff = abs(y_pred - st.session_state.fp_y_true)
    status = "SUCCESS" if diff < 0.05 else "ADJUSTING"
    st.markdown(f"""
    <div style="background:rgba(10, 10, 20, 0.7); backdrop-filter: blur(12px); border:1px solid rgba(139,92,246,0.3); border-radius:16px; padding:30px; box-shadow:0 16px 40px rgba(0,0,0,0.5); text-align:center; word-wrap:break-word; position:relative; overflow:hidden;">
        <div style="position: absolute; right: -10%; top: -50%; width: 200px; height: 200px; background: radial-gradient(circle, rgba(0,240,255,0.1) 0%, transparent 70%); border-radius: 50%;"></div>
        <div style="font-family:'Oswald', sans-serif; font-size:20px; color:#a78bfa; letter-spacing:2px; font-weight:700; text-transform:uppercase;">NETWORK ACCURACY PROTOCOL</div>
        <div style="font-size:72px; font-family:'Oswald', sans-serif; font-weight:700; color:{'#00f0ff' if status=='SUCCESS' else '#FAFAFA'}; filter:drop-shadow(0 0 15px rgba(0,240,255,0.4));">
            {100 - (diff*100):.1f}% MATCH
        </div>
        <div style="font-weight:400; font-family:'Inter'; color:#E4E4E7; margin-top:8px; text-transform:uppercase; letter-spacing:2px;">STATUS: <span style="color:{'#00f0ff' if status=='SUCCESS' else '#a78bfa'}; font-weight:700;">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Network diagram with values
    diag_vals = [st.session_state.fp_Xv] + \
        [As[i+1].flatten().tolist() for i in range(len(s_sizes)-2)] + [[y_pred]]
    if len(s_sizes) <= MAX_LAYERS:
        st.plotly_chart(draw_network(s_sizes, s_lbl, vals=diag_vals),
            use_container_width=True)

    # Activation Heatmap
    heat = _activation_heatmap(As, s_lbl)
    if heat: st.plotly_chart(heat, use_container_width=True, theme=None)

    # Layer tabs
    tabs = st.tabs([f"L{i} — {l}" for i, l in enumerate(s_lbl[1:], 1)] + ["📊 Loss Analysis"])
    for li, tab in enumerate(tabs[:-1]):
        with tab:
            Z = Zs[li]; Act = As[li+1]; is_o = li == len(Zs)-1
            act_name = o_act if is_o else st.session_state.fp_h_acts[li]
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:{MUTED};margin-bottom:12px;">Z = W·A_prev + b  →  A = {act_name}(Z)</div>""",
                unsafe_allow_html=True)
            if is_o:
                c1, c2 = st.columns(2)
                c1.metric("Pre-activation Z", f"{Z[0,0]:.6f}")
                c2.metric("Output ŷ", f"{Act[0,0]:.6f}")
            else:
                nn = Z.shape[0]; cols = st.columns(min(nn, 5))
                for j in range(nn):
                    cols[j%5].metric(f"Z{j+1}", f"{Z[j,0]:.4f}")
                    cols[j%5].metric(f"A{j+1}", f"{Act[j,0]:.4f}")
    with tabs[-1]:
        c1, c2 = st.columns(2)
        fig_b = go.Figure([
            go.Bar(name="ŷ", x=["Value"], y=[y_pred], marker_color="#06B6D4",
                text=[f"{y_pred:.4f}"], textposition="auto"),
            go.Bar(name="y", x=["Value"], y=[st.session_state.fp_y_true],
                marker_color="#10B981", text=[f"{st.session_state.fp_y_true:.4f}"],
                textposition="auto"),
        ])
        fig_b.update_layout(barmode="group",
            title=dict(text="Prediction vs Target", font=dict(color=TEXT, family="Impact", size=16)),
            **plotly_layout())
        c1.plotly_chart(fig_b, use_container_width=True, theme=None)
