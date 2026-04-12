import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.chatbot import push_tutor_insight, render_chatbot
from utils.learning_ui import (
    contribution_bar,
    heatmap_with_text,
    line_story_chart,
    render_ai_coach_panel,
    render_learning_journey,
    render_story_card,
    render_summary_panel,
    render_step_grid,
    render_visualization_mode,
    scatter3d_story,
)
from utils.nlp_engine import generate_fwd_insight
from utils.nn_helpers import (
    ACTS,
    LOSSES,
    A,
    C,
    G,
    R,
    draw_network,
    forward_pass,
    make_weights,
    plotly_layout,
)
from utils.styles import gradient_header, inject_global_css, render_log, section_header, speedometer, inject_module_theme, MODULE_THEMES
from utils.voice import render_voice_button

MAX_NODES = 6
MAX_LAYERS = 7


def _wkey(n_in, hidden_sizes):
    return f"fp_w_{n_in}_{'_'.join(str(x) for x in hidden_sizes)}"


def _get_w(n_in, hidden_sizes):
    key = _wkey(n_in, hidden_sizes)
    if key not in st.session_state:
        st.session_state[key] = make_weights(n_in, hidden_sizes)
    return st.session_state[key]


def _act_curve_fig(act_name):
    xs = np.linspace(-6, 6, 300)
    fn = ACTS[act_name]["fn"]
    ys = fn(xs)
    xs2 = xs + 0.001
    deriv = (fn(xs2) - fn(xs)) / 0.001
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{act_name}",
            line=dict(color="#EF4444", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=np.clip(deriv, -5, 5),
            mode="lines",
            name="Approx derivative",
            line=dict(color="#3B82F6", width=2, dash="dot"),
        )
    )
    fig.add_hline(y=0, line=dict(color="#334155", width=1))
    fig.add_vline(x=0, line=dict(color="#334155", width=1))
    fig.update_layout(
        title=dict(text=f"{act_name} Activation", font=dict(color="#FFFFFF", family="Montserrat", size=18)),
        **plotly_layout(
            xaxis=dict(title="z", color="#94A3B8", gridcolor="#334155", range=[-6, 6]),
            yaxis=dict(title="f(z)", color="#94A3B8", gridcolor="#334155"),
            height=260,
            margin=dict(l=40, r=20, t=55, b=35),
        ),
    )
    return fig


def _matrix_for_heatmap(arrays, row_labels, title):
    rows = []
    for arr in arrays:
        flat = np.array(arr, dtype=float).flatten()[:8]
        rows.append(list(flat) + [np.nan] * (8 - len(flat)))
    return heatmap_with_text(rows, [f"n{i+1}" for i in range(8)], row_labels, title, height=300)


def _layer_script(layer_label, act_name, A_prev, Z, A_cur):
    return (
        f"{layer_label} is receiving {A_prev.shape[0]} signals from the previous layer. "
        f"Each neuron forms a weighted sum, producing pre-activation values like {np.round(Z.flatten()[:4], 3).tolist()}. "
        f"Then the {act_name} activation reshapes those scores into outputs like {np.round(A_cur.flatten()[:4], 3).tolist()}. "
        "That transformed signal becomes the input for the next layer."
    )


def _layer_summary_table(Zs, As, labels, h_acts, o_act):
    rows = []
    for idx, (Z, A_cur) in enumerate(zip(Zs, As[1:])):
        act_name = o_act if idx == len(Zs) - 1 else h_acts[idx]
        rows.append(
            {
                "Layer": labels[idx + 1],
                "Activation": act_name,
                "Neurons": int(A_cur.shape[0]),
                "mean(Z)": float(np.mean(Z)),
                "std(Z)": float(np.std(Z)),
                "mean(A)": float(np.mean(A_cur)),
                "max(A)": float(np.max(A_cur)),
            }
        )
    return pd.DataFrame(rows)


def _layer_contribution_chart(weights, As, labels, layer_idx, neuron_idx):
    W, b = weights[layer_idx]
    A_prev = As[layer_idx].flatten()
    neuron_idx = max(0, min(neuron_idx, W.shape[0] - 1))
    contribs = (W[neuron_idx] * A_prev).astype(float)
    prev_labels = [f"{labels[layer_idx]} {i+1}" for i in range(len(A_prev))]
    return contribution_bar(
        prev_labels + ["bias"],
        list(contribs) + [float(b[neuron_idx, 0])],
        f"Contribution Breakdown For {labels[layer_idx + 1]} neuron {neuron_idx + 1}",
        positive=G,
        negative=R,
        neutral=A,
        y_title="Contribution to z",
        height=320,
    )


def _prediction_audit_fig(y_pred, y_true):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Prediction", "Target"],
            y=[y_pred, y_true],
            marker_color=[C, G],
            text=[f"{y_pred:.4f}", f"{y_true:.4f}"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="Prediction vs Target", font=dict(color="#FFFFFF", family="Montserrat", size=18)),
        **plotly_layout(
            height=300,
            yaxis=dict(title="Value", color="#94A3B8", gridcolor="#334155"),
            xaxis=dict(color="#94A3B8", gridcolor="#334155"),
            margin=dict(l=40, r=20, t=55, b=35),
        ),
    )
    return fig


def _prepare_ai_explanation(result):
    if result.get("ai_attempted"):
        return result

    from utils.ai_helper import get_ai_explanation

    prompt = (
        "Explain a neural network forward pass to a complete beginner using friendly everyday language. "
        f"The network architecture was {result['arch_text']}. "
        f"The final output prediction was {result['y_pred']:.6f}, the target was {result['y_true']:.6f}, "
        f"and the loss using {result['loss_fn']} was {result['loss']:.6f}. "
        "Walk layer by layer through Z = W dot A_prev + b, explain what an activation function does, "
        "and clarify why deeper layers transform the signal instead of just copying it."
    )

    with st.spinner("Generating beginner-friendly AI lesson..."):
        ai_text = get_ai_explanation(
            prompt,
            system_prompt=(
                "You are an unforgettable AI tutor for beginners. "
                "Write 5 to 7 short paragraphs with clear analogies, gentle pacing, and concrete step-by-step explanations."
            ),
            max_tokens=420,
        )

    fallback = generate_fwd_insight(
        result["h_acts"][-1] if result["h_acts"] else "Linear",
        result["loss_fn"],
        result["loss"],
    )
    result["ai_text"] = ai_text or fallback
    result["ai_label"] = "AI Tutor // Forward Signal Narrator" if ai_text else "Forward Insight"
    result["ai_attempted"] = True
    if not result.get("ai_pushed"):
        push_tutor_insight(result["ai_text"], result["ai_label"])
        result["ai_pushed"] = True
    return result


def _activation_3d_story(result):
    xs, ys, zs, texts = [], [], [], []
    for layer_idx, activations in enumerate(result["As"][1:], start=1):
        flat = activations.flatten()
        for neuron_idx, value in enumerate(flat, start=1):
            xs.append(layer_idx)
            ys.append(neuron_idx)
            zs.append(float(value))
            texts.append(
                f"{result['labels'][layer_idx]} neuron {neuron_idx}<br>activation={float(value):+.4f}"
            )

    return scatter3d_story(
        [
            {
                "name": "Activation cloud",
                "x": xs,
                "y": ys,
                "z": zs,
                "mode": "markers",
                "size": 6,
                "color_values": zs,
                "text": texts,
                "hovertemplate": "%{text}<extra></extra>",
            }
        ],
        "3D Activation Landscape",
        "Layer",
        "Neuron",
        "Activation",
        height=430,
    )


def forward_propagation_page():
    inject_global_css()
    inject_module_theme("forward_prop")
    gradient_header(
        "Forward Propagation",
        "Animated Layer Walkthrough · Watch signals turn into features, features into activations, and activations into predictions",
        "➡️",
    )

    theory_text = (
        "Forward propagation is the act of pushing information through a neural network. "
        "Every layer builds a weighted sum, applies an activation function, and hands the transformed signal to the next layer."
    )
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown(
            """
            **The Forward Pass**

            1. Compute the linear part: `Z = W·A_prev + b`
            2. Apply a non-linear activation: `A = f(Z)`
            3. Repeat until the output layer produces a prediction `ŷ`
            4. Compare `ŷ` with the target `y` using a loss function
            """
        )
        render_voice_button(theory_text, key_suffix="fp_theory")

    render_learning_journey(
        "Turn A Raw Input Vector Into A Decision",
        "This page makes the forward pass feel tangible. Instead of a single hidden calculation, you will see each layer reshape the signal in sequence.",
        [
            "Weights decide which parts of the incoming signal matter most.",
            "Bias lets a neuron shift its baseline before activating.",
            "Activation functions bend or squash the signal so the network can learn richer patterns.",
            "The output layer converts all earlier transformations into the final prediction.",
        ],
        "Imagine a package moving through a series of smart checkpoints. Each checkpoint reads the package, stamps it with new information, and sends a more refined package onward.",
        audio_text=theory_text,
        key_suffix="fp_intro",
    )

    if "fp_result" not in st.session_state:
        st.session_state.fp_result = None

    st.divider()
    section_header("1. Network Architecture Builder", "Design the shape of your feed-forward network")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        n_in = c1.slider("Input features", 1, 20, 3)
        n_hid = c2.slider("Hidden layers", 1, 6, 2)
        same = c3.checkbox("Uniform hidden width", True)

        hidden_sizes = []
        if same:
            width = st.slider("Neurons per hidden layer", 1, 20, 4)
            hidden_sizes = [width] * n_hid
        else:
            cols = st.columns(min(n_hid, 5))
            for idx in range(n_hid):
                hidden_sizes.append(cols[idx % 5].slider(f"H{idx+1}", 1, 20, 3, key=f"fp_h_{idx}"))

        sizes = [n_in] + hidden_sizes + [1]
        labels = ["Input"] + [f"H{idx+1}" for idx in range(n_hid)] + ["Output"]
        arch_text = " → ".join([f"{label}({size})" for label, size in zip(labels, sizes)])
        st.markdown(" → ".join([f"**{label}**({size})" for label, size in zip(labels, sizes)]))

    if len(sizes) <= MAX_LAYERS:
        st.plotly_chart(draw_network(sizes, labels), use_container_width=True, key="fp_arch_net")

    st.divider()
    section_header("2. Inputs & Target", "Choose the signal entering the network")
    with st.container(border=True):
        input_ready = True
        source = st.radio("Input source", ["Single Sample", "CSV Upload", "Manual Table"], horizontal=True, key="fp_input_source")
        Xv = [round(0.25 + idx * 0.18, 2) for idx in range(n_in)]
        y_true = 1.0

        if source == "Single Sample":
            cols = st.columns(min(n_in + 1, 5))
            for idx in range(n_in):
                Xv[idx] = cols[idx % 4].number_input(f"x{idx+1}", value=Xv[idx], step=0.1, key=f"fp_x_{idx}")
            y_true = cols[min(n_in, 4)].number_input("Target y", value=1.0, step=0.1, key="fp_target")
        elif source == "CSV Upload":
            input_ready = False
            upload_col, preview_col = st.columns([1.1, 1.3])
            with upload_col:
                uploaded = st.file_uploader("Upload CSV", type=["csv"], key="fp_dataset_upload")
                if uploaded is not None:
                    df = pd.read_csv(uploaded)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= n_in + 1:
                        default_features = numeric_cols[:n_in]
                        feature_cols = st.multiselect("Feature columns", numeric_cols, default=default_features, key="fp_feature_cols")
                        if len(feature_cols) == n_in:
                            target_choices = [col for col in numeric_cols if col not in feature_cols] or numeric_cols
                            target_col = st.selectbox("Target column", target_choices, index=0, key="fp_target_col")
                            proc = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
                            if not proc.empty:
                                row_idx = st.number_input("Sample row", min_value=0, max_value=len(proc) - 1, value=0, step=1, key="fp_row_idx")
                                sample = proc.iloc[int(row_idx)]
                                Xv = [float(sample[col]) for col in feature_cols]
                                y_true = float(sample[target_col])
                                input_ready = True
                            else:
                                st.warning("The uploaded dataset has no complete rows after processing.")
                        else:
                            st.warning(f"Select exactly {n_in} feature columns.")
                    else:
                        st.warning(f"The uploaded dataset needs at least {n_in + 1} numeric columns.")
            with preview_col:
                if input_ready:
                    st.dataframe(pd.DataFrame([Xv + [y_true]], columns=[f"x{i+1}" for i in range(n_in)] + ["target"]).round(4), use_container_width=True, hide_index=True)
                else:
                    st.info("Upload a dataset and choose feature columns to process a sample row.")
        else:
            input_ready = False
            default_rows = pd.DataFrame([{**{f"x{i+1}": round(0.25 + i * 0.18, 2) for i in range(n_in)}, "target": 1.0} for _ in range(3)])
            edited = st.data_editor(default_rows, num_rows="dynamic", use_container_width=True, key=f"fp_manual_table_{n_in}")
            proc = edited.dropna().reset_index(drop=True)
            if not proc.empty:
                row_idx = st.number_input("Sample row", min_value=0, max_value=len(proc) - 1, value=0, step=1, key="fp_manual_row_idx")
                sample = proc.iloc[int(row_idx)]
                Xv = [float(sample[f"x{i+1}"]) for i in range(n_in)]
                y_true = float(sample["target"])
                input_ready = True
                st.dataframe(proc.round(4), use_container_width=True, hide_index=True)
            else:
                st.warning("Add at least one full row to the manual table.")

        X = np.array(Xv, dtype=float).reshape(-1, 1)

    st.divider()
    section_header("3. Activations, Loss, and Walkthrough Controls", "Configure the logic used in each layer")
    a1, a2, a3, a4 = st.columns(4)
    same_act = a1.checkbox("Same hidden activation", True)
    if same_act:
        act_all = a2.selectbox("Hidden activation", list(ACTS.keys()))
        h_acts = [act_all] * n_hid
        with st.expander("Activation curve", expanded=False):
            st.plotly_chart(_act_curve_fig(act_all), use_container_width=True, key="fp_act_curve")
    else:
        h_acts = []
        cols = st.columns(min(n_hid, 5))
        for idx in range(n_hid):
            h_acts.append(cols[idx % 5].selectbox(f"H{idx+1} activation", list(ACTS.keys()), key=f"fp_act_{idx}"))
    o_act = a3.selectbox("Output activation", list(ACTS.keys()), index=1)
    loss_fn = a4.selectbox("Loss function", list(LOSSES.keys()))

    c1, c2 = st.columns(2)
    animate = c1.checkbox("Animate layer walkthrough", True)
    delay = c2.slider("Walkthrough delay (s)", 0.0, 0.8, 0.18, 0.02)

    st.divider()
    section_header("4. Weight Management", "Randomize or manually set weights and biases")
    can_manual = n_in <= 6 and all(h <= 6 for h in hidden_sizes)
    mode = st.radio("Weight mode", ["Random", "Manual"], horizontal=True) if can_manual else "Random"
    if not can_manual:
        st.warning("Manual entry is disabled for larger networks to keep the interface manageable.")

    weights = _get_w(n_in, hidden_sizes)
    if mode == "Random":
        if st.button("🎲 Randomize Weights", use_container_width=True):
            st.session_state[_wkey(n_in, hidden_sizes)] = make_weights(n_in, hidden_sizes)
            st.rerun()
        weights = _get_w(n_in, hidden_sizes)
        with st.expander("View current weights", expanded=False):
            for layer_idx, (W, b) in enumerate(weights):
                layer_name = "Output" if layer_idx == len(weights) - 1 else f"Hidden {layer_idx + 1}"
                df = pd.DataFrame(W, columns=[f"in{i+1}" for i in range(W.shape[1])], index=[f"n{j+1}" for j in range(W.shape[0])])
                df["bias"] = b.flatten()
                st.caption(layer_name)
                st.dataframe(df.round(4), use_container_width=True)
    else:
        manual_weights = []
        incoming = n_in
        for layer_idx, width in enumerate(hidden_sizes):
            Wm = np.zeros((width, incoming))
            bm = np.zeros((width, 1))
            with st.expander(f"Hidden {layer_idx + 1} W({width}×{incoming})", expanded=layer_idx == 0):
                for j in range(width):
                    row = st.columns(incoming + 1)
                    for i in range(incoming):
                        Wm[j, i] = row[i].number_input(
                            f"W[{j+1},{i+1}]",
                            value=0.5 if i == j else 0.0,
                            min_value=-2.0,
                            max_value=2.0,
                            step=0.1,
                            key=f"fp_mw_{layer_idx}_{j}_{i}",
                        )
                    bm[j, 0] = row[incoming].number_input(
                        f"b[{j+1}]",
                        value=0.0,
                        min_value=-2.0,
                        max_value=2.0,
                        step=0.1,
                        key=f"fp_mb_{layer_idx}_{j}",
                    )
            manual_weights.append((Wm, bm))
            incoming = width

        Wo = np.zeros((1, incoming))
        bo = np.zeros((1, 1))
        with st.expander(f"Output W(1×{incoming})", expanded=True):
            row = st.columns(incoming + 1)
            for i in range(incoming):
                Wo[0, i] = row[i].number_input(
                    f"Wo[{i+1}]",
                    value=1.0,
                    min_value=-2.0,
                    max_value=2.0,
                    step=0.1,
                    key=f"fp_out_w_{i}",
                )
            bo[0, 0] = row[incoming].number_input(
                "bo",
                value=0.0,
                min_value=-2.0,
                max_value=2.0,
                step=0.1,
                key="fp_out_b",
            )
        manual_weights.append((Wo, bo))
        weights = manual_weights

    st.divider()
    reset_col, run_col = st.columns([1, 4])
    if reset_col.button("Reset", use_container_width=True):
        for key in [k for k in list(st.session_state.keys()) if k.startswith("fp_")]:
            del st.session_state[key]
        st.rerun()
    run_btn = run_col.button("▶ Run Forward Pass", type="primary", use_container_width=True)

    log_placeholder = st.expander("📋 Computation Log", expanded=False).empty()

    if run_btn:
        if not input_ready:
            st.warning("Process a valid input row first.")
            render_chatbot(
        "forward propagation and layer-by-layer signal transmission",
        system_prompt=(
            "You are a calm, methodical signal-processing engineer. "
            "You explain forward propagation step by step, tracing how data flows through each layer. "
            "You use precise technical language but always connect math to intuition."
        ),
        greeting=(
            "➡️ Signal Engineer online. I trace how data flows through neural networks layer by layer. "
            "Ask me about weighted sums, activation functions, loss computation, or how to read the layer heatmaps."
        ),
        theme=MODULE_THEMES["forward_prop"],
        tutor_label="SIGNAL ENGINEER ➡️",
        placeholder="Ask about forward propagation or signal flow...",
    )
            return
        Zs, As = forward_pass(X, weights, h_acts, o_act)
        y_pred = float(As[-1][0, 0])
        loss = float(LOSSES[loss_fn]["fn"](As[-1], np.array([[y_true]], dtype=float)))

        log_lines = [f"Inputs: {[round(v, 4) for v in Xv]}"]
        for layer_idx, (W, b) in enumerate(weights):
            layer_name = labels[layer_idx + 1]
            act_name = o_act if layer_idx == len(weights) - 1 else h_acts[layer_idx]
            log_lines.append(f"{layer_name}: Z = W·A_prev + b, then A = {act_name}(Z)")
            log_lines.append(f"Z sample: {np.round(Zs[layer_idx].flatten()[:5], 4).tolist()}")
            log_lines.append(f"A sample: {np.round(As[layer_idx + 1].flatten()[:5], 4).tolist()}")
        log_lines.append(f"Prediction ŷ = {y_pred:.6f}, target y = {y_true:.6f}, loss = {loss:.6f}")
        render_log(log_placeholder, log_lines)

        if animate:
            walk_holder = st.empty()
            for layer_idx, (Z, A_cur) in enumerate(zip(Zs, As[1:])):
                act_name = o_act if layer_idx == len(Zs) - 1 else h_acts[layer_idx]
                layer_name = labels[layer_idx + 1]
                neuron_idx = int(np.argmax(np.abs(Z.flatten())))
                layer_script = _layer_script(layer_name, act_name, As[layer_idx], Z, A_cur)
                diag_vals = [Xv] + [As[i + 1].flatten().tolist() for i in range(len(sizes) - 2)] + [[y_pred]]
                with walk_holder.container():
                    render_step_grid(
                        [
                            {
                                "eyebrow": "Active Layer",
                                "title": layer_name,
                                "value": act_name,
                                "caption": "This is the layer currently transforming the signal.",
                                "accent": C,
                            },
                            {
                                "eyebrow": "Linear Step",
                                "title": "mean(Z)",
                                "value": f"{np.mean(Z):+.3f}",
                                "caption": "Weighted sums are built before activation happens.",
                                "accent": G,
                            },
                            {
                                "eyebrow": "Activation Step",
                                "title": "mean(A)",
                                "value": f"{np.mean(A_cur):+.3f}",
                                "caption": "The activation function reshapes the raw scores.",
                                "accent": A,
                            },
                            {
                                "eyebrow": "Most Excited Neuron",
                                "title": f"Neuron {neuron_idx + 1}",
                                "value": f"{Z.flatten()[neuron_idx]:+.3f}",
                                "caption": "This neuron received the strongest raw signal in the layer.",
                                "accent": R if Z.flatten()[neuron_idx] < 0 else G,
                            },
                        ],
                        columns=4,
                    )
                    render_story_card(
                        f"{layer_name} is reshaping the signal",
                        layer_script,
                        eyebrow="Live Layer Narration",
                        accent=C,
                        key_suffix=f"fp_walk_{layer_idx}",
                    )
                    c1, c2 = st.columns([1.2, 1])
                    if len(sizes) <= MAX_LAYERS:
                        c1.plotly_chart(
                            draw_network(sizes, labels, vals=diag_vals, highlight=layer_idx + 1),
                            use_container_width=True,
                            key=f"fp_walk_net_{layer_idx}",
                        )
                    c2.plotly_chart(
                        _layer_contribution_chart(weights, As, labels, layer_idx, neuron_idx),
                        use_container_width=True,
                        key=f"fp_walk_contrib_{layer_idx}",
                    )
                if delay > 0:
                    time.sleep(delay)
            walk_holder.empty()

        st.session_state.fp_result = {
            "sizes": sizes,
            "labels": labels,
            "Xv": Xv,
            "X": X,
            "y_true": float(y_true),
            "weights": weights,
            "Zs": Zs,
            "As": As,
            "loss": loss,
            "y_pred": y_pred,
            "loss_fn": loss_fn,
            "h_acts": h_acts,
            "o_act": o_act,
            "arch_text": arch_text,
            "summary_df": _layer_summary_table(Zs, As, labels, h_acts, o_act),
            "ai_attempted": False,
            "ai_pushed": False,
        }

    result = st.session_state.get("fp_result")
    if not result:
        render_chatbot(
        "forward propagation and layer-by-layer signal transmission",
        system_prompt=(
            "You are a calm, methodical signal-processing engineer. "
            "You explain forward propagation step by step, tracing how data flows through each layer. "
            "You use precise technical language but always connect math to intuition."
        ),
        greeting=(
            "➡️ Signal Engineer online. I trace how data flows through neural networks layer by layer. "
            "Ask me about weighted sums, activation functions, loss computation, or how to read the layer heatmaps."
        ),
        theme=MODULE_THEMES["forward_prop"],
        tutor_label="SIGNAL ENGINEER ➡️",
        placeholder="Ask about forward propagation or signal flow...",
    )
        return

    result = _prepare_ai_explanation(result)
    st.session_state.fp_result = result

    st.divider()
    section_header("5. AI Lesson + Audio", "A narrated explanation of the complete forward pass")
    render_story_card(
        "What this forward pass is doing",
        result["ai_text"],
        eyebrow=result["ai_label"],
        accent=C,
        key_suffix="fp_ai_story",
    )

    diff = abs(result["y_pred"] - result["y_true"])
    match_score = max(0.0, min(100.0, 100.0 - diff * 100.0))
    pred_mag_max = max(1.0, abs(result["y_pred"]), abs(result["y_true"])) + 1.0
    diag_vals = [result["Xv"]] + [result["As"][i + 1].flatten().tolist() for i in range(len(result["sizes"]) - 2)] + [[result["y_pred"]]]

    view_mode = render_visualization_mode("fp", accent=C, subject="forward propagation")

    render_summary_panel(
        "Forward-pass snapshot",
        [
            f"Architecture: `{result['arch_text']}`.",
            f"The network turned `{len(result['Xv'])}` input features into a prediction of `{result['y_pred']:.6f}`.",
            f"The target was `{result['y_true']:.6f}`, so the absolute error is `{diff:.6f}` and the loss is `{result['loss']:.6f}`.",
            "Use `Layer Explorer` to zoom into one layer and see how its neurons reshape the signal.",
        ],
        eyebrow="WHAT TO NOTICE",
        accent=C,
    )

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(speedometer(abs(result["y_pred"]), pred_mag_max, "Pred Magnitude", color=C, height=220), use_container_width=True, key="fp_g_pred")
    g2.plotly_chart(speedometer(min(result["loss"], 2.0), 2.0, "Loss", color=R, height=220), use_container_width=True, key="fp_g_loss")
    g3.plotly_chart(speedometer(match_score, 100, "Match", color=G, height=220), use_container_width=True, key="fp_g_match")

    coach_narratives = {
        "Big Picture": (
            "Forward propagation is just signal transformation. Each layer receives a vector, mixes it with weights and bias, then reshapes it through an activation function."
        ),
        "Layer Math": (
            "Inside one layer, every neuron computes its own weighted sum. That means the layer is really many small judges all reading the same incoming activations from different angles."
        ),
        "Activation Intuition": (
            "Without activation functions, stacked layers collapse into one big linear transformation. Non-linearity is what lets the network bend and sculpt the signal into richer features."
        ),
        "Debugging View": (
            "If the output looks wrong, inspect mean(Z), mean(A), and the contribution chart. These show whether a layer is amplifying, suppressing, or saturating the signal."
        ),
    }
    if view_mode == "Immersive Coach":
        render_ai_coach_panel("Coach focus", coach_narratives, key_suffix="fp_focus", accent=C)
    elif view_mode == "3D Visualization Explorer":
        st.plotly_chart(_activation_3d_story(result), use_container_width=True, key="fp_3d_story_live")

    st.divider()
    section_header("6. Explore The Signal", "Start with the overall flow, then zoom into one layer only when you want more detail")
    overview_tab, layer_tab, audit_tab = st.tabs(["Overview", "Layer Explorer", "Audit & Tables"])

    with overview_tab:
        render_step_grid(
            [
                {
                    "eyebrow": "Output",
                    "title": "Prediction ŷ",
                    "value": f"{result['y_pred']:.6f}",
                    "caption": "This is the final value produced by the network.",
                    "accent": C,
                },
                {
                    "eyebrow": "Reference",
                    "title": "Target y",
                    "value": f"{result['y_true']:.6f}",
                    "caption": "This is the answer you wanted the network to produce.",
                    "accent": G,
                },
                {
                    "eyebrow": "Distance",
                    "title": "Absolute Error",
                    "value": f"{diff:.6f}",
                    "caption": "Smaller means the prediction landed closer to the target.",
                    "accent": A,
                },
                {
                    "eyebrow": "Loss",
                    "title": result["loss_fn"],
                    "value": f"{result['loss']:.6f}",
                    "caption": "The loss function compresses prediction quality into one scalar score.",
                    "accent": R,
                },
            ],
            columns=4,
        )
        c1, c2 = st.columns([1.15, 0.85])
        with c1:
            if len(result["sizes"]) <= MAX_LAYERS:
                st.plotly_chart(
                    draw_network(result["sizes"], result["labels"], vals=diag_vals),
                    use_container_width=True,
                    key="fp_final_net",
                )
            else:
                st.info("The network is a little too deep to draw cleanly here, so use the charts below to inspect how the signal changes layer by layer.")
        with c2:
            st.plotly_chart(_prediction_audit_fig(result["y_pred"], result["y_true"]), use_container_width=True, key="fp_overview_audit")
        st.plotly_chart(
            line_story_chart(
                [
                    {"name": "mean(Z)", "x": list(range(1, len(result["Zs"]) + 1)), "y": [float(np.mean(z)) for z in result["Zs"]], "color": C},
                    {"name": "mean(A)", "x": list(range(1, len(result["As"]))), "y": [float(np.mean(a)) for a in result["As"][1:]], "color": G},
                ],
                "How The Signal Changes Across Layers",
                "Average value",
                height=320,
            ),
            use_container_width=True,
            key="fp_signal_story_overview",
        )

    with layer_tab:
        focused_tab, atlas_tab = st.tabs(["Focused Layer", "Activation Atlas"])

        with focused_tab:
            layer_options = [f"{idx + 1}. {label}" for idx, label in enumerate(result["labels"][1:])]
            layer_choice = st.selectbox("Focus layer", layer_options, key="fp_focus_layer")
            layer_idx = layer_options.index(layer_choice)
            layer_name = result["labels"][layer_idx + 1]
            act_name = result["o_act"] if layer_idx == len(result["Zs"]) - 1 else result["h_acts"][layer_idx]
            Z = result["Zs"][layer_idx]
            A_cur = result["As"][layer_idx + 1]
            layer_script = _layer_script(layer_name, act_name, result["As"][layer_idx], Z, A_cur)
            render_step_grid(
                [
                    {
                        "eyebrow": "Layer",
                        "title": layer_name,
                        "value": act_name,
                        "caption": "This activation function defines how the layer reshapes its raw scores.",
                        "accent": C,
                    },
                    {
                        "eyebrow": "Pre-Activation",
                        "title": "mean(Z)",
                        "value": f"{np.mean(Z):+.4f}",
                        "caption": "Average raw score before activation is applied.",
                        "accent": G,
                    },
                    {
                        "eyebrow": "Post-Activation",
                        "title": "mean(A)",
                        "value": f"{np.mean(A_cur):+.4f}",
                        "caption": "Average outgoing signal after non-linearity.",
                        "accent": A,
                    },
                    {
                        "eyebrow": "Width",
                        "title": "Neurons",
                        "value": str(A_cur.shape[0]),
                        "caption": "This layer contains this many simultaneous feature detectors.",
                        "accent": R,
                    },
                ],
                columns=4,
            )
            render_story_card(
                f"How {layer_name} transforms the signal",
                layer_script,
                eyebrow="Layer Explanation",
                accent=C,
                key_suffix=f"fp_focus_audio_{layer_idx}",
            )

            neuron_idx = st.slider(
                "Neuron to inspect",
                min_value=1,
                max_value=int(result["weights"][layer_idx][0].shape[0]),
                value=1,
                key="fp_focus_neuron",
            )
            st.plotly_chart(
                _layer_contribution_chart(result["weights"], result["As"], result["labels"], layer_idx, neuron_idx - 1),
                use_container_width=True,
                key="fp_focus_contrib",
            )

        with atlas_tab:
            st.plotly_chart(
                _matrix_for_heatmap(result["Zs"], result["labels"][1:], "Pre-Activation Map (Z)"),
                use_container_width=True,
                key="fp_z_heatmap",
            )
            st.plotly_chart(
                _matrix_for_heatmap(result["As"][1:], result["labels"][1:], "Activation Map (A)"),
                use_container_width=True,
                key="fp_a_heatmap",
            )

    with audit_tab:
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            contribution_bar(
                ["Prediction - Target"],
                [result["y_pred"] - result["y_true"]],
                "Signed Prediction Delta",
                positive=G,
                negative=R,
                neutral=A,
                y_title="Delta",
                height=300,
            ),
            use_container_width=True,
            key="fp_delta_bar",
        )
        c2.plotly_chart(_prediction_audit_fig(result["y_pred"], result["y_true"]), use_container_width=True, key="fp_audit_bar_full")
        with st.expander("Open the layer summary table", expanded=False):
            st.dataframe(result["summary_df"].round(4), use_container_width=True, hide_index=True)
            st.caption("This summary shows how the signal distribution changes from layer to layer.")

    render_chatbot(
        "forward propagation and layer-by-layer signal transmission",
        system_prompt=(
            "You are a calm, methodical signal-processing engineer. "
            "You explain forward propagation step by step, tracing how data flows through each layer. "
            "You use precise technical language but always connect math to intuition."
        ),
        greeting=(
            "➡️ Signal Engineer online. I trace how data flows through neural networks layer by layer. "
            "Ask me about weighted sums, activation functions, loss computation, or how to read the layer heatmaps."
        ),
        theme=MODULE_THEMES["forward_prop"],
        tutor_label="SIGNAL ENGINEER ➡️",
        placeholder="Ask about forward propagation or signal flow...",
    )
if __name__ == "__main__":
    forward_propagation_page()
