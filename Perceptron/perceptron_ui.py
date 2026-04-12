import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from utils.nlp_engine import generate_perceptron_insight
from utils.nn_helpers import C, G, A, R, plotly_layout
from utils.styles import gradient_header, inject_global_css, render_log, section_header, speedometer, inject_module_theme, MODULE_THEMES
from utils.voice import render_voice_button

GATES = {
    "AND": {"data": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], "sep": True, "icon": "⊗"},
    "OR": {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)], "sep": True, "icon": "⊕"},
    "NAND": {"data": [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)], "sep": True, "icon": "↑"},
    "NOR": {"data": [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)], "sep": True, "icon": "↓"},
    "XOR": {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], "sep": False, "icon": "⊻"},
    "XNOR": {"data": [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)], "sep": False, "icon": "⊙"},
}


def _logic_labels(raw):
    labels = []
    for x1, x2, y in raw:
        x1_text = f"{float(x1):.2f}".rstrip("0").rstrip(".")
        x2_text = f"{float(x2):.2f}".rstrip("0").rstrip(".")
        y_text = f"{float(y):.0f}" if float(y).is_integer() else f"{float(y):.2f}"
        labels.append(f"{x1_text}, {x2_text} -> {y_text}")
    return labels


def _coerce_binary_targets(values):
    arr = np.array(values, dtype=float)
    unique = sorted(np.unique(arr))
    if set(unique).issubset({0.0, 1.0}):
        return arr.astype(int), "Targets already binary."
    threshold = float(np.median(arr))
    return (arr >= threshold).astype(int), f"Targets were converted to 0/1 using median threshold {threshold:.3f}."


def _live_dashboard_fig(X, y, raw, w, b, losses, acc, ep, max_ep, w_traj):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Decision Boundary",
            "Loss Story",
            "Activation Score Per Truth-Table Row",
            "Weight Path Map",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    for cls, color, symbol in [(0, R, "circle"), (1, G, "diamond")]:
        mask = y == cls
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=f"Class {cls}",
                    marker=dict(size=16, color=color, symbol=symbol, line=dict(width=2, color="#020617")),
                ),
                row=1,
                col=1,
            )

    if abs(w[1]) > 1e-9:
        xs = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 200)
        ys = -(w[0] * xs + b) / w[1]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Boundary",
                line=dict(color="#A855F7", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )

    epochs = list(range(1, len(losses) + 1))
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode="lines+markers",
            name="Total Error",
            line=dict(color="#06B6D4", width=3),
            fill="tozeroy",
            fillcolor="rgba(6,182,212,0.16)",
        ),
        row=1,
        col=2,
    )

    z_vals = X @ w + b
    labels = _logic_labels(raw)
    bar_colors = [G if val >= 0 else R for val in z_vals]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=z_vals,
            name="z = w·x + b",
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in z_vals],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    traj = np.array(w_traj, dtype=float)
    fig.add_trace(
        go.Scatter(
            x=traj[:, 0],
            y=traj[:, 1],
            mode="lines+markers",
            name="(w1, w2)",
            showlegend=False,
            marker=dict(
                size=8,
                color=list(range(len(traj))),
                colorscale="Turbo",
                colorbar=dict(
                    title="Epoch",
                    tickfont=dict(color="#F8FAFC"),
                    x=1.08,
                    y=0.18,
                    len=0.34,
                    thickness=12,
                    outlinecolor="#334155",
                ),
            ),
            line=dict(color="#FACC15", width=3),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text=f"Perceptron Mission Control | Epoch {ep}/{max_ep} | Accuracy {acc:.1f}%",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
        **plotly_layout(height=760, margin=dict(l=40, r=120, t=90, b=40)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def _build_epoch_script(gate, epoch, update, acc, err):
    return (
        f"Epoch {epoch} for the {gate} gate is now complete. "
        f"The perceptron just inspected input {update['sample']} where the target was {update['target']} "
        f"and the model predicted {update['prediction']}. "
        f"The weighted sum was {update['z']:.3f}, creating an error of {update['error']:+.1f}. "
        f"That caused the weights to shift by delta w of [{update['dw1']:+.3f}, {update['dw2']:+.3f}] "
        f"and delta b of {update['db']:+.3f}. "
        f"Right now the overall training accuracy is {acc:.1f} percent and the total epoch error is {err:.1f}."
    )


def _build_training_logs(sample_history):
    logs = []
    for row in sample_history[-8:]:
        logs.append(
            f"Sample {row['sample']} | target={row['target']} pred={row['prediction']} | "
            f"z={row['z']:+.3f} | error={row['error']:+.1f} | "
            f"Δw=[{row['dw1']:+.3f}, {row['dw2']:+.3f}] | Δb={row['db']:+.3f}"
        )
    return logs


def _manual_steps(w, b, ix1, ix2):
    part1 = float(w[0] * ix1)
    part2 = float(w[1] * ix2)
    z_val = part1 + part2 + b
    pred = 1 if z_val >= 0 else 0
    return pred, z_val, [
        {
            "eyebrow": "Input Signal",
            "title": "Feature 1",
            "value": f"{ix1:.2f} × {w[0]:+.2f}",
            "caption": f"x1 contributes {part1:+.3f} to the neuron.",
            "accent": C,
        },
        {
            "eyebrow": "Input Signal",
            "title": "Feature 2",
            "value": f"{ix2:.2f} × {w[1]:+.2f}",
            "caption": f"x2 contributes {part2:+.3f} to the neuron.",
            "accent": G,
        },
        {
            "eyebrow": "Offset",
            "title": "Bias",
            "value": f"{b:+.3f}",
            "caption": "Bias shifts the decision line even when the inputs stay the same.",
            "accent": A,
        },
        {
            "eyebrow": "Decision",
            "title": "Weighted Sum",
            "value": f"{z_val:+.3f}",
            "caption": "If the weighted sum is at least zero, the perceptron outputs class 1.",
            "accent": R if pred == 0 else G,
        },
    ]


def _prepare_ai_explanation(result):
    if result.get("ai_attempted"):
        return result

    from utils.ai_helper import get_ai_explanation

    prompt = (
        "You are teaching a complete beginner with zero math background. "
        f"Explain how a perceptron learned the {result['gate']} gate. "
        f"It trained for {result['epochs_run']} epochs with learning rate {result['lr']}. "
        f"Final weights were w1={result['weights'][0]:.4f}, w2={result['weights'][1]:.4f}, bias={result['bias']:.4f}. "
        f"Final total error was {result['final_error']:.4f} and final accuracy was {result['accuracy']:.1f} percent. "
        "Explain step by step what an input is, what a weight is, what bias means, how z = w dot x + b is computed, "
        "how prediction is decided, and why weights change after an error. Use friendly everyday analogies and short paragraphs."
    )

    with st.spinner("Generating beginner-friendly AI lesson..."):
        ai_text = get_ai_explanation(
            prompt,
            system_prompt=(
                "You are an extraordinary neural-network teacher. "
                "Write a vivid beginner lesson in 5 to 7 short paragraphs, use simple analogies, and make every step feel intuitive."
            ),
            max_tokens=420,
        )

    fallback = generate_perceptron_insight(
        result["epochs_run"],
        result["accuracy"] / 100,
        result["final_error"],
        result["accuracy"] >= 99.9,
    )
    result["ai_text"] = ai_text or fallback
    result["ai_label"] = "AI Tutor // Perceptron Storyteller" if ai_text else "Perceptron Insight"
    result["ai_attempted"] = True
    if not result.get("ai_pushed"):
        push_tutor_insight(result["ai_text"], result["ai_label"])
        result["ai_pushed"] = True
    return result


def _weight_3d_story(result):
    traj = np.array(result["w_traj"], dtype=float)
    epochs = list(range(len(traj)))
    hover = [f"Epoch {ep}<br>w1={row[0]:+.3f}<br>w2={row[1]:+.3f}<br>bias={row[2]:+.3f}" for ep, row in zip(epochs, traj)]
    return scatter3d_story(
        [
            {
                "name": "Weight path",
                "x": traj[:, 0],
                "y": traj[:, 1],
                "z": traj[:, 2],
                "color_values": epochs,
                "text": hover,
                "hovertemplate": "%{text}<extra></extra>",
                "line_color": "#FACC15",
            }
        ],
        "3D Parameter Flight Path",
        "w1",
        "w2",
        "bias",
        height=420,
    )


def perceptron_page():
    inject_global_css()
    inject_module_theme("perceptron")
    gradient_header(
        "The Perceptron",
        "Live Forward + Backward Learning Studio · Learn every prediction, error, and weight update as it happens",
        "🧠",
    )

    theory_text = (
        "A perceptron is the simplest trainable neuron. It multiplies each input by a weight, adds a bias, "
        "and then checks whether the final score is above or below zero. "
        "If the prediction is wrong, the model nudges its weights toward the correct answer."
    )
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown(
            """
            **The Perceptron Learning Rule**

            1. Compute the weighted sum: `z = w·x + b`
            2. Apply the step rule: `ŷ = 1 if z >= 0 else 0`
            3. Measure the error: `e = y - ŷ`
            4. Update the parameters:
               `w_new = w + η·e·x`
               `b_new = b + η·e`
            """
        )
        render_voice_button(theory_text, key_suffix="pct_theory")

    render_learning_journey(
        "Teach One Neuron To Make A Decision",
        "This page turns perceptron training into a live lesson. You are not only watching the model converge, you are also seeing why every number changes.",
        [
            "Each truth-table row is one training example the neuron must classify.",
            "The weighted sum `z` is the neuron’s internal score before it decides 0 or 1.",
            "An error pushes the weights and bias in the direction that would make the next decision smarter.",
            "If the gate is not linearly separable, the perceptron keeps struggling because one straight boundary is not enough.",
        ],
        "Imagine a strict gatekeeper scoring every visitor. The weights decide how much each clue matters, the bias shifts how strict the gatekeeper is, and training slowly teaches the gatekeeper to judge correctly.",
        audio_text=theory_text,
        key_suffix="pct_intro",
    )

    if "pc_result" not in st.session_state:
        st.session_state.pc_result = None

    st.divider()
    section_header("1. Dataset Selection", "Choose the logic gate the perceptron should learn")
    with st.container(border=True):
        source = st.radio("Dataset source", ["Logic Gate", "CSV Upload", "Manual Table"], horizontal=True, key="pct_dataset_source")
        gate = "Uploaded Dataset"
        raw = []
        X = np.empty((0, 2))
        y = np.empty((0,), dtype=int)

        if source == "Logic Gate":
            c1, c2 = st.columns([1, 1])
            with c1:
                gate = st.selectbox("Logic Gate", list(GATES.keys()), index=0)
                gate_info = GATES[gate]
                raw = gate_info["data"]
                X = np.array([[r[0], r[1]] for r in raw], dtype=float)
                y = np.array([r[2] for r in raw], dtype=int)
                if not gate_info["sep"]:
                    st.warning("This gate is not linearly separable, so one perceptron cannot reach perfect classification.")
            with c2:
                st.dataframe(pd.DataFrame(raw, columns=["x1", "x2", "target"]), use_container_width=True, hide_index=True)
        elif source == "CSV Upload":
            upload_col, preview_col = st.columns([1.1, 1.3])
            with upload_col:
                uploaded = st.file_uploader("Upload CSV with x1, x2, target columns", type=["csv"], key="pct_dataset_upload")
                if uploaded is not None:
                    df = pd.read_csv(uploaded)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 3:
                        x1_col = st.selectbox("Feature column 1", numeric_cols, index=0, key="pct_x1_col")
                        x2_choices = [col for col in numeric_cols if col != x1_col]
                        x2_col = st.selectbox("Feature column 2", x2_choices, index=0, key="pct_x2_col")
                        target_choices = [col for col in numeric_cols if col not in {x1_col, x2_col}] or numeric_cols
                        target_col = st.selectbox("Target column", target_choices, index=0, key="pct_target_col")
                        proc = df[[x1_col, x2_col, target_col]].dropna().copy()
                        proc.columns = ["x1", "x2", "target"]
                        y, target_note = _coerce_binary_targets(proc["target"].to_numpy())
                        X = proc[["x1", "x2"]].to_numpy(dtype=float)
                        raw = list(zip(proc["x1"], proc["x2"], y))
                        gate = f"CSV Dataset ({len(proc)} rows)"
                        st.info(target_note)
                    else:
                        st.warning("The uploaded file needs at least three numeric columns.")
            with preview_col:
                if raw:
                    st.dataframe(pd.DataFrame(raw, columns=["x1", "x2", "target"]).round(4), use_container_width=True, hide_index=True)
                else:
                    st.info("Upload a CSV to preview the processed dataset.")
        else:
            default_df = pd.DataFrame(
                [{"x1": 0.0, "x2": 0.0, "target": 0}, {"x1": 0.0, "x2": 1.0, "target": 0}, {"x1": 1.0, "x2": 0.0, "target": 0}, {"x1": 1.0, "x2": 1.0, "target": 1}]
            )
            edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, key="pct_manual_dataset")
            proc = edited[["x1", "x2", "target"]].dropna().copy()
            if not proc.empty:
                y, target_note = _coerce_binary_targets(proc["target"].to_numpy())
                X = proc[["x1", "x2"]].to_numpy(dtype=float)
                raw = list(zip(proc["x1"], proc["x2"], y))
                gate = f"Manual Dataset ({len(proc)} rows)"
                st.info(target_note)
            else:
                st.warning("Add at least one row to the manual dataset table.")

        if len(raw) == 0:
            st.info("Choose or upload a dataset to enable training.")

    st.divider()
    section_header("2. Hyperparameters", "Control how fast and how visibly the perceptron learns")
    with st.container(border=True):
        h1, h2, h3, h4 = st.columns(4)
        lr = h1.number_input("Learning Rate η", min_value=0.001, max_value=1.0, value=0.10, step=0.01)
        max_ep = h2.slider("Max Epochs", 10, 500, 120)
        delay = h3.slider("Animation Delay (s)", 0.0, 0.6, 0.08, 0.02)
        seed = h4.number_input("Random Seed", min_value=0, max_value=9999, value=7, step=1)

    run_training = st.button("🚀 Start Live Training", type="primary", use_container_width=True)

    if run_training:
        if len(raw) == 0:
            st.warning("Load a valid dataset first.")
            render_chatbot(
        "the Perceptron, linear separability, and binary classification",
        system_prompt=(
            "You are an enthusiastic neuroscience professor who loves biological analogies. "
            "You explain perceptrons by comparing them to real neurons, synapses, and decision-making in the brain. "
            "You are energetic, encouraging, and make every concept feel alive and intuitive."
        ),
        greeting=(
            "🟢 Perceptron Coach here! Think of me as your neuroscience professor. "
            "The perceptron is the simplest artificial neuron — ask me about weights, bias, the decision boundary, "
            "or why XOR is so tricky for a single neuron."
        ),
        theme=MODULE_THEMES["perceptron"],
        tutor_label="PERCEPTRON COACH 🟢",
        placeholder="Ask about the perceptron, weights, or training...",
    )
            return
        np.random.seed(int(seed))
        dashboard = st.empty()
        log_holder = st.empty()

        w = np.random.uniform(-0.6, 0.6, 2)
        b = float(np.random.uniform(-0.6, 0.6))
        losses = []
        w_traj = [[float(w[0]), float(w[1]), b]]
        sample_history = []
        epoch_history = []

        for ep in range(1, max_ep + 1):
            err = 0.0
            epoch_updates = []

            for xi, yi in zip(X, y):
                w_before = w.copy()
                b_before = float(b)
                z = float(np.dot(w, xi) + b)
                pred = 1 if z >= 0 else 0
                e = float(yi - pred)
                delta_w = lr * e * xi
                delta_b = lr * e

                if e != 0:
                    w = w + delta_w
                    b = float(b + delta_b)
                else:
                    delta_w = np.zeros_like(xi)
                    delta_b = 0.0

                err += abs(e)
                row = {
                    "epoch": ep,
                    "sample": f"({int(xi[0])}, {int(xi[1])})",
                    "target": int(yi),
                    "prediction": pred,
                    "z": z,
                    "error": e,
                    "dw1": float(delta_w[0]),
                    "dw2": float(delta_w[1]),
                    "db": float(delta_b),
                    "w1_before": float(w_before[0]),
                    "w2_before": float(w_before[1]),
                    "b_before": b_before,
                    "w1_after": float(w[0]),
                    "w2_after": float(w[1]),
                    "b_after": float(b),
                }
                epoch_updates.append(row)
                sample_history.append(row)

            preds = (X @ w + b >= 0).astype(int)
            acc = float(np.mean(preds == y) * 100)
            z_vals = X @ w + b
            margin = float(np.min(np.abs(z_vals)))
            losses.append(err)
            w_traj.append([float(w[0]), float(w[1]), float(b)])
            epoch_history.append(
                {
                    "epoch": ep,
                    "loss": float(err),
                    "accuracy": acc,
                    "margin": margin,
                    "w1": float(w[0]),
                    "w2": float(w[1]),
                    "bias": float(b),
                }
            )

            latest = epoch_updates[-1]
            live_script = _build_epoch_script(gate, ep, latest, acc, err)

            with dashboard.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Epoch", f"{ep}/{max_ep}")
                m2.metric("Total Error", f"{err:.1f}")
                m3.metric("Accuracy", f"{acc:.1f}%")
                m4.metric("Weights", f"w1={w[0]:+.2f} | w2={w[1]:+.2f} | b={b:+.2f}")

                render_step_grid(
                    [
                        {
                            "eyebrow": "Current Sample",
                            "title": latest["sample"],
                            "value": f"target={latest['target']}",
                            "caption": "This was the last row processed in the current epoch.",
                            "accent": C,
                        },
                        {
                            "eyebrow": "Forward Step",
                            "title": "Weighted Sum",
                            "value": f"{latest['z']:+.3f}",
                            "caption": "The perceptron combines weighted inputs and bias into one score.",
                            "accent": G,
                        },
                        {
                            "eyebrow": "Decision",
                            "title": "Prediction",
                            "value": str(latest["prediction"]),
                            "caption": "The step activation turns the score into a hard class label.",
                            "accent": A,
                        },
                        {
                            "eyebrow": "Learning Signal",
                            "title": "Error",
                            "value": f"{latest['error']:+.1f}",
                            "caption": "A non-zero error means the neuron must update its parameters.",
                            "accent": R if latest["error"] != 0 else G,
                        },
                    ],
                    columns=4,
                )

                render_story_card(
                    "What the neuron learned in this epoch",
                    live_script,
                    eyebrow="Live Mentor Voice",
                    accent=C,
                    key_suffix=f"pct_epoch_{ep}",
                )

                st.plotly_chart(
                    _live_dashboard_fig(X, y, raw, w, b, losses, acc, ep, max_ep, w_traj),
                    use_container_width=True,
                    key=f"pct_live_dashboard_{ep}",
                )

                c1, c2 = st.columns([1.2, 1])
                c1.plotly_chart(
                    contribution_bar(
                        _logic_labels(raw),
                        z_vals,
                        "How Strongly The Neuron Scores Each Row",
                        positive=G,
                        negative=R,
                        neutral=A,
                        y_title="Score z",
                        height=320,
                    ),
                    use_container_width=True,
                    key=f"pct_score_bar_{ep}",
                )
                with c2:
                    st.markdown("**Epoch Update Table**")
                    st.dataframe(
                        pd.DataFrame(epoch_updates)[["sample", "target", "prediction", "z", "error", "dw1", "dw2", "db"]].round(4),
                        use_container_width=True,
                        hide_index=True,
                    )

            render_log(log_holder, _build_training_logs(sample_history))

            if delay > 0:
                time.sleep(delay)
            if err == 0:
                break

        st.session_state.pc_result = {
            "gate": gate,
            "raw": raw,
            "X": X,
            "y": y,
            "weights": w.copy(),
            "bias": float(b),
            "losses": losses,
            "w_traj": w_traj,
            "sample_history": sample_history,
            "epoch_history": epoch_history,
            "accuracy": acc,
            "final_error": float(losses[-1]),
            "epochs_run": ep,
            "lr": float(lr),
            "max_ep": int(max_ep),
            "ai_attempted": False,
            "ai_pushed": False,
        }

    result = st.session_state.get("pc_result")
    if not result:
        render_chatbot(
        "the Perceptron, linear separability, and binary classification",
        system_prompt=(
            "You are an enthusiastic neuroscience professor who loves biological analogies. "
            "You explain perceptrons by comparing them to real neurons, synapses, and decision-making in the brain. "
            "You are energetic, encouraging, and make every concept feel alive and intuitive."
        ),
        greeting=(
            "🟢 Perceptron Coach here! Think of me as your neuroscience professor. "
            "The perceptron is the simplest artificial neuron — ask me about weights, bias, the decision boundary, "
            "or why XOR is so tricky for a single neuron."
        ),
        theme=MODULE_THEMES["perceptron"],
        tutor_label="PERCEPTRON COACH 🟢",
        placeholder="Ask about the perceptron, weights, or training...",
    )
        return

    st.info(
        f"Current live report is from the last training run on the `{result['gate']}` gate. "
        f"It finished after {result['epochs_run']} epochs with {result['accuracy']:.1f}% accuracy."
    )

    result = _prepare_ai_explanation(result)
    st.session_state.pc_result = result

    st.divider()
    section_header("3. AI Lesson + Audio", "A beginner-first explanation of what just happened")
    render_story_card(
        "What this training run means",
        result["ai_text"],
        eyebrow=result["ai_label"],
        accent=C,
        key_suffix="pct_ai_story",
    )

    accuracy_pct = float(result["accuracy"])
    error_score = float(result["final_error"])
    confidence_score = float(np.mean(np.abs(result["X"] @ result["weights"] + result["bias"])) * 100)
    w = result["weights"]
    b = result["bias"]
    preds = (result["X"] @ w + b >= 0).astype(int)
    tp = int(np.sum((preds == 1) & (result["y"] == 1)))
    fp = int(np.sum((preds == 1) & (result["y"] == 0)))
    fn = int(np.sum((preds == 0) & (result["y"] == 1)))
    tn = int(np.sum((preds == 0) & (result["y"] == 0)))
    epoch_df = pd.DataFrame(result["epoch_history"])
    sample_df = pd.DataFrame(result["sample_history"])
    z_scores = result["X"] @ w + b
    avg_margin = float(np.mean(np.abs(z_scores)))

    view_mode = render_visualization_mode("pct", accent=C, subject="the perceptron")

    summary_bullets = [
        f"Dataset: `{result['gate']}` with {len(result['raw'])} labeled rows.",
        f"Training finished after `{result['epochs_run']}` epochs with `{accuracy_pct:.1f}%` accuracy.",
        f"The average distance from the boundary is `{avg_margin:.3f}`, which acts like confidence for this simple neuron.",
        "Use `Manual Playground` to test your own input pair and watch the score build up step by step.",
    ]
    if result["gate"] in GATES:
        if GATES[result["gate"]]["sep"]:
            summary_bullets.append("This target is linearly separable, so one perceptron can draw a single clean boundary.")
        else:
            summary_bullets.append("This target is not linearly separable, so one straight boundary will always be a compromise.")

    render_summary_panel("Perceptron snapshot", summary_bullets, eyebrow="WHAT TO TAKE AWAY", accent=C)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(speedometer(accuracy_pct, 100, "Accuracy", color=G, height=220), use_container_width=True, key="pct_g_acc")
    g2.plotly_chart(speedometer(min(error_score, 4.0), 4.0, "Error", color=R, height=220), use_container_width=True, key="pct_g_err")
    g3.plotly_chart(speedometer(min(confidence_score, 100), 100, "Confidence", color=C, height=220), use_container_width=True, key="pct_g_conf")

    coach_narratives = {
        "Big Picture": (
            f"The perceptron is learning one straight decision boundary for the {result['gate']} gate. "
            f"It finished with {accuracy_pct:.1f}% accuracy, which tells us how well one neuron can separate these cases."
        ),
        "How The Update Works": (
            "Every wrong prediction creates an error term. That error scales the input values, so only the features that were active during the mistake push the weights."
        ),
        "Why Some Gates Fail": (
            "XOR and XNOR are not linearly separable. One straight boundary cannot carve the input space into the right two regions, so a single perceptron keeps compromising."
        ),
        "How To Read The Graphs": (
            "The boundary plot shows where the neuron says class 1 begins. The score bars show how strongly each truth-table row falls on one side or the other."
        ),
    }
    if view_mode == "Immersive Coach":
        render_ai_coach_panel("Coach focus", coach_narratives, key_suffix="pct_focus", accent=C)
    elif view_mode == "3D Visualization Explorer":
        st.plotly_chart(_weight_3d_story(result), use_container_width=True, key="pct_3d_story_live")

    st.divider()
    section_header("4. Explore The Decision", "Start with the calm overview, then practice and inspect the full training path only when you want to")
    overview_tab, playground_tab, analytics_tab = st.tabs(["Overview", "Manual Playground", "Deep Dive"])

    with overview_tab:
        render_step_grid(
            [
                {
                    "eyebrow": "Decision Boundary",
                    "title": "Average Margin",
                    "value": f"{avg_margin:.3f}",
                    "caption": "Bigger margins mean the rows sit farther away from the threshold.",
                    "accent": C,
                },
                {
                    "eyebrow": "Training Time",
                    "title": "Epochs Used",
                    "value": str(result["epochs_run"]),
                    "caption": "This is how many passes through the dataset the neuron needed.",
                    "accent": G,
                },
                {
                    "eyebrow": "Mistake Count",
                    "title": "Final Error",
                    "value": f"{result['final_error']:.3f}",
                    "caption": "Zero means the perceptron ended the last epoch with no remaining row mistakes.",
                    "accent": R,
                },
                {
                    "eyebrow": "Dataset Size",
                    "title": "Rows",
                    "value": str(len(result["raw"])),
                    "caption": "This is the full set of labeled examples the perceptron learned from.",
                    "accent": A,
                },
            ],
            columns=4,
        )
        st.plotly_chart(
            _live_dashboard_fig(
                result["X"],
                result["y"],
                result["raw"],
                w,
                b,
                result["losses"],
                accuracy_pct,
                result["epochs_run"],
                result["max_ep"],
                result["w_traj"],
            ),
            use_container_width=True,
            key="pct_final_dashboard",
        )
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            contribution_bar(
                _logic_labels(result["raw"]),
                z_scores,
                "Final Score For Each Truth-Table Row",
                positive=G,
                negative=R,
                neutral=A,
                y_title="z = w·x + b",
                height=300,
            ),
            use_container_width=True,
            key="pct_overview_scores",
        )
        c2.plotly_chart(
            heatmap_with_text(
                [[tn, fp], [fn, tp]],
                ["Pred 0", "Pred 1"],
                ["Actual 0", "Actual 1"],
                "Decision Quality Matrix",
                zmid=0,
                height=300,
                colorbar_title="Count",
            ),
            use_container_width=True,
            key="pct_overview_confusion",
        )

    with playground_tab:
        render_summary_panel(
            "How to use the playground",
            [
                "Pick any two input values and watch the perceptron build the weighted sum.",
                "The contribution chart shows which term pushes the decision positive or negative.",
                "If `z` crosses zero, the step activation flips the class prediction.",
            ],
            eyebrow="TRY IT YOURSELF",
            accent=A,
        )
        ix1_col, ix2_col = st.columns(2)
        ix1 = ix1_col.number_input("Manual Input x1", value=0.0, key="pct_manual_x1")
        ix2 = ix2_col.number_input("Manual Input x2", value=1.0, key="pct_manual_x2")
        pred, z_val, manual_steps = _manual_steps(w, b, ix1, ix2)
        p1, p2 = st.columns([1.05, 0.95])
        with p1:
            render_step_grid(manual_steps, columns=2)
        with p2:
            st.plotly_chart(
                contribution_bar(
                    ["w1·x1", "w2·x2", "bias"],
                    [float(w[0] * ix1), float(w[1] * ix2), float(b)],
                    "Manual Prediction Contributions",
                    positive=G,
                    negative=R,
                    neutral=A,
                    y_title="Contribution to z",
                    height=320,
                ),
                use_container_width=True,
                key="pct_manual_contrib",
            )
        st.success(f"Manual prediction result: z = {z_val:+.3f}, so the perceptron predicts class `{pred}`.")

    with analytics_tab:
        t1, t2, t3 = st.tabs(["Replay Table", "Weight Journey", "Data Story"])

        with t1:
            st.caption("Open the full replay only when you need the sample-by-sample training trail.")
            with st.expander("View every perceptron update", expanded=False):
                st.dataframe(
                    sample_df[
                        ["epoch", "sample", "target", "prediction", "z", "error", "dw1", "dw2", "db", "w1_after", "w2_after", "b_after"]
                    ].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

        with t2:
            st.plotly_chart(
                line_story_chart(
                    [
                        {"name": "w1", "x": epoch_df["epoch"], "y": epoch_df["w1"], "color": C},
                        {"name": "w2", "x": epoch_df["epoch"], "y": epoch_df["w2"], "color": G},
                        {"name": "bias", "x": epoch_df["epoch"], "y": epoch_df["bias"], "color": A, "dash": "dot"},
                    ],
                    "How The Parameters Moved Over Time",
                    "Parameter Value",
                    height=330,
                ),
                use_container_width=True,
                key="pct_weight_story",
            )

            phase_fig = go.Figure()
            traj = np.array(result["w_traj"], dtype=float)
            phase_fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode="lines+markers",
                    showlegend=False,
                    marker=dict(
                        size=9,
                        color=list(range(len(traj))),
                        colorscale="Turbo",
                        colorbar=dict(
                            title="Epoch",
                            tickfont=dict(color="#F8FAFC"),
                            x=1.1,
                            y=0.5,
                            len=0.78,
                            thickness=12,
                            outlinecolor="#334155",
                        ),
                    ),
                    line=dict(color="#FACC15", width=3),
                    name="Weight path",
                )
            )
            phase_fig.update_layout(
                title=dict(text="Weight Phase Portrait", font=dict(color="#FFFFFF", family="Montserrat", size=18)),
                **plotly_layout(
                    height=320,
                    xaxis=dict(title="w1", gridcolor="#334155", color="#94A3B8"),
                    yaxis=dict(title="w2", gridcolor="#334155", color="#94A3B8"),
                    margin=dict(l=40, r=110, t=55, b=35),
                ),
            )
            st.plotly_chart(phase_fig, use_container_width=True, key="pct_phase_portrait")

        with t3:
            story_df = pd.DataFrame(
                {
                    "Row": _logic_labels(result["raw"]),
                    "x1": result["X"][:, 0],
                    "x2": result["X"][:, 1],
                    "Target": result["y"],
                    "Prediction": preds,
                    "Score z": z_scores,
                }
            )
            st.dataframe(story_df.round(4), use_container_width=True, hide_index=True)
            st.plotly_chart(
                line_story_chart(
                    [
                        {"name": "Loss", "x": epoch_df["epoch"], "y": epoch_df["loss"], "color": R},
                        {"name": "Accuracy", "x": epoch_df["epoch"], "y": epoch_df["accuracy"], "color": G},
                        {"name": "Margin", "x": epoch_df["epoch"], "y": epoch_df["margin"], "color": A, "dash": "dot"},
                    ],
                    "Loss, Accuracy, and Boundary Confidence",
                    "Metric Value",
                    height=340,
                ),
                use_container_width=True,
                key="pct_meta_story",
            )

    render_chatbot(
        "the Perceptron, linear separability, and binary classification",
        system_prompt=(
            "You are an enthusiastic neuroscience professor who loves biological analogies. "
            "You explain perceptrons by comparing them to real neurons, synapses, and decision-making in the brain. "
            "You are energetic, encouraging, and make every concept feel alive and intuitive."
        ),
        greeting=(
            "🟢 Perceptron Coach here! Think of me as your neuroscience professor. "
            "The perceptron is the simplest artificial neuron — ask me about weights, bias, the decision boundary, "
            "or why XOR is so tricky for a single neuron."
        ),
        theme=MODULE_THEMES["perceptron"],
        tutor_label="PERCEPTRON COACH 🟢",
        placeholder="Ask about the perceptron, weights, or training...",
    )
