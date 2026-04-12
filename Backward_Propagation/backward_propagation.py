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
from utils.nlp_engine import generate_bwd_insight
from utils.nn_helpers import ACTS, LOSSES, C, G, A, R, backward_pass, draw_network, forward_pass, make_weights, plotly_layout
from utils.styles import gradient_header, inject_global_css, render_log, section_header, speedometer, inject_module_theme, MODULE_THEMES
from utils.voice import render_voice_button

MAX_LAYERS = 7


def _gradient_chain_heatmap(grads, labels, lr):
    rows = []
    for grad in grads:
        rows.append(
            [
                float(np.mean(np.abs(grad["dLdA"]))),
                float(np.mean(np.abs(grad["dAdZ"]))),
                float(np.mean(np.abs(grad["dLdZ"]))),
                float(np.mean(np.abs(grad["dLdW"]))),
                float(np.mean(np.abs(lr * grad["dLdW"]))),
            ]
        )
    return heatmap_with_text(
        rows,
        ["|dL/dA|", "|dA/dZ|", "|dL/dZ|", "|dL/dW|", "|ΔW|"],
        labels,
        "Chain Rule Signal Strength",
        zmid=0,
        height=320,
        colorbar_title="Mean abs",
    )


def _layer_update_table(grads, labels, lr):
    rows = []
    for layer_name, grad in zip(labels, grads):
        rows.append(
            {
                "Layer": layer_name,
                "mean|dL/dA|": float(np.mean(np.abs(grad["dLdA"]))),
                "mean|dA/dZ|": float(np.mean(np.abs(grad["dAdZ"]))),
                "mean|dL/dW|": float(np.mean(np.abs(grad["dLdW"]))),
                "mean|ΔW|": float(np.mean(np.abs(lr * grad["dLdW"]))),
                "mean|Δb|": float(np.mean(np.abs(lr * grad["dLdb"]))),
            }
        )
    return pd.DataFrame(rows)


def _epoch_script(epoch, y_pred, y_true, loss, mean_grad, strongest_layer):
    return (
        f"Epoch {epoch} is complete. The network predicted {y_pred:.4f} while the target was {y_true:.4f}, "
        f"so the current loss is {loss:.5f}. The average gradient magnitude is {mean_grad:.5f}, and the strongest learning signal is flowing through {strongest_layer}. "
        "Backpropagation is using the chain rule to trace responsibility backward and then adjust the weights in the opposite direction of the error."
    )


def _weight_delta_heatmap(before, after, title):
    W_before, _ = before
    W_after, _ = after
    delta = np.array(W_after - W_before, dtype=float)
    return heatmap_with_text(
        delta,
        [f"in{i+1}" for i in range(delta.shape[1])],
        [f"n{j+1}" for j in range(delta.shape[0])],
        title,
        zmid=0,
        height=340,
        colorbar_title="ΔW",
    )


def _gradient_matrix_heatmap(grad, title):
    matrix = np.array(grad["dLdW"], dtype=float)
    return heatmap_with_text(
        matrix,
        [f"in{i+1}" for i in range(matrix.shape[1])],
        [f"n{j+1}" for j in range(matrix.shape[0])],
        title,
        zmid=0,
        height=340,
        colorbar_title="dL/dW",
    )


def _layer_gradient_story(layer_names, history, title, color_pool):
    series = []
    x = list(range(1, len(history) + 1))
    for idx, layer_name in enumerate(layer_names):
        series.append(
            {
                "name": layer_name,
                "x": x,
                "y": [epoch[idx] for epoch in history],
                "color": color_pool[idx % len(color_pool)],
            }
        )
    return line_story_chart(series, title, "Mean absolute value", height=330)


def _prepare_ai_explanation(result):
    if result.get("ai_attempted"):
        return result

    from utils.ai_helper import get_ai_explanation

    prompt = (
        "Teach backpropagation to a complete beginner. "
        f"The network architecture was {result['arch_text']}. "
        f"It trained for {result['epochs_run']} epochs with learning rate {result['lr']}, using {result['loss_fn']} loss. "
        f"The final prediction was {result['y_pred']:.6f}, the target was {result['y_true']:.6f}, the final loss was {result['loss']:.6f}, "
        f"and the average gradient magnitude was {result['mean_grad']:.6f}. "
        "Explain the chain rule, gradients, weight updates, and why the sign and size of the gradient matter. "
        "Use simple analogies and make the explanation feel live and intuitive."
    )

    with st.spinner("Generating beginner-friendly AI lesson..."):
        ai_text = get_ai_explanation(
            prompt,
            system_prompt=(
                "You are an extraordinary deep-learning tutor for beginners. "
                "Write 5 to 7 short paragraphs that patiently explain each step with vivid analogies and no assumed prior knowledge."
            ),
            max_tokens=430,
        )

    fallback = generate_bwd_insight(result["epochs_run"], result["max_ep"], result["loss"], result["mean_grad"])
    result["ai_text"] = ai_text or fallback
    result["ai_label"] = "AI Tutor // Backpropagation Storyteller" if ai_text else "Backpropagation Insight"
    result["ai_attempted"] = True
    if not result.get("ai_pushed"):
        push_tutor_insight(result["ai_text"], result["ai_label"])
        result["ai_pushed"] = True
    return result


def _gradient_3d_story(result):
    xs, ys, zs, texts = [], [], [], []
    layer_names = result["labels"][1:]
    for epoch_idx, layer_values in enumerate(result["layer_grad_history"], start=1):
        for layer_idx, value in enumerate(layer_values, start=1):
            xs.append(epoch_idx)
            ys.append(layer_idx)
            zs.append(float(value))
            texts.append(f"Epoch {epoch_idx}<br>{layer_names[layer_idx - 1]}<br>mean|dL/dW|={float(value):.6f}")

    return scatter3d_story(
        [
            {
                "name": "Gradient field",
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
        "3D Gradient Flow",
        "Epoch",
        "Layer",
        "Gradient magnitude",
        height=430,
    )


def backward_propagation_page():
    inject_global_css()
    inject_module_theme("backward_prop")
    gradient_header(
        "Backward Propagation",
        "Live Chain Rule Lab · See error signals travel backward, gradients form, and weights update in real time",
        "⬅️",
    )

    theory_text = (
        "Backward propagation teaches the network after it makes a prediction. "
        "The loss measures how wrong the prediction is, the chain rule tracks how each weight contributed to that mistake, "
        "and gradient descent updates the weights to reduce future error."
    )
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown(
            """
            **The Backward Pass**

            1. Compute the prediction error from the loss function
            2. Use the chain rule to move that learning signal backward:
               `dL/dW = dL/dA × dA/dZ × dZ/dW`
            3. Update each parameter with gradient descent:
               `W_new = W - η·dL/dW`
               `b_new = b - η·dL/db`
            """
        )
        render_voice_button(theory_text, key_suffix="bp_theory")

    render_learning_journey(
        "Teach The Network By Sending The Mistake Backward",
        "This page makes backpropagation visible. You will watch the loss create a teaching signal, see how strongly each layer receives it, and inspect how the weights change because of it.",
        [
            "The loss tells the model how wrong it was, but not yet how to fix itself.",
            "Gradients measure how much each weight influenced that mistake.",
            "The chain rule links the layers together so blame can move backward through the whole network.",
            "Gradient descent uses those blame signals to nudge weights in the direction that should lower the loss next time.",
        ],
        "Think of a relay race played in reverse. The final runner discovers the team lost time, then sends that information backward through every earlier runner so each one knows how much to adjust.",
        audio_text=theory_text,
        key_suffix="bp_intro",
    )

    if "bp_result" not in st.session_state:
        st.session_state.bp_result = None

    st.divider()
    section_header("1. Network Architecture", "Define the model that will learn through backpropagation")
    c1, c2, c3 = st.columns(3)
    n_in = c1.slider("Input features", 1, 10, 2)
    n_hid = c2.slider("Hidden layers", 1, 4, 1)
    seed = c3.number_input("Random Seed", min_value=0, max_value=9999, value=13, step=1)

    hidden_sizes = []
    cols = st.columns(min(n_hid, 4))
    for idx in range(n_hid):
        hidden_sizes.append(cols[idx].slider(f"H{idx+1} Width", 1, 10, 3, key=f"bp_h_{idx}"))
    sizes = [n_in] + hidden_sizes + [1]
    labels = ["Input"] + [f"H{idx+1}" for idx in range(n_hid)] + ["Output"]
    arch_text = " → ".join([f"{label}({size})" for label, size in zip(labels, sizes)])
    st.markdown(" → ".join([f"**{label}**({size})" for label, size in zip(labels, sizes)]))

    st.divider()
    section_header("2. Inputs & Target", "Set the training example that will generate the gradients")
    with st.container(border=True):
        input_ready = True
        source = st.radio("Input source", ["Single Sample", "CSV Upload", "Manual Table"], horizontal=True, key="bp_input_source")
        Xv = [round(0.3 + idx * 0.25, 2) for idx in range(n_in)]
        y_true = 1.0

        if source == "Single Sample":
            cols = st.columns(min(n_in + 1, 5))
            for idx in range(n_in):
                Xv[idx] = cols[idx % 4].number_input(f"x{idx+1}", value=Xv[idx], step=0.1, key=f"bp_x_{idx}")
            y_true = cols[min(n_in, 4)].number_input("Target y", value=1.0, step=0.1, key="bp_target")
        elif source == "CSV Upload":
            input_ready = False
            upload_col, preview_col = st.columns([1.1, 1.3])
            with upload_col:
                uploaded = st.file_uploader("Upload CSV", type=["csv"], key="bp_dataset_upload")
                if uploaded is not None:
                    df = pd.read_csv(uploaded)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= n_in + 1:
                        feature_cols = st.multiselect("Feature columns", numeric_cols, default=numeric_cols[:n_in], key="bp_feature_cols")
                        if len(feature_cols) == n_in:
                            target_choices = [col for col in numeric_cols if col not in feature_cols] or numeric_cols
                            target_col = st.selectbox("Target column", target_choices, index=0, key="bp_target_col")
                            proc = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
                            if not proc.empty:
                                row_idx = st.number_input("Sample row", min_value=0, max_value=len(proc) - 1, value=0, step=1, key="bp_row_idx")
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
            default_rows = pd.DataFrame([{**{f"x{i+1}": round(0.3 + i * 0.25, 2) for i in range(n_in)}, "target": 1.0} for _ in range(3)])
            edited = st.data_editor(default_rows, num_rows="dynamic", use_container_width=True, key=f"bp_manual_table_{n_in}")
            proc = edited.dropna().reset_index(drop=True)
            if not proc.empty:
                row_idx = st.number_input("Sample row", min_value=0, max_value=len(proc) - 1, value=0, step=1, key="bp_manual_row_idx")
                sample = proc.iloc[int(row_idx)]
                Xv = [float(sample[f"x{i+1}"]) for i in range(n_in)]
                y_true = float(sample["target"])
                input_ready = True
                st.dataframe(proc.round(4), use_container_width=True, hide_index=True)
            else:
                st.warning("Add at least one full row to the manual table.")
        X = np.array(Xv, dtype=float).reshape(-1, 1)

    st.divider()
    section_header("3. Hyperparameters & Activations", "Decide how the network learns")
    h1, h2, h3, h4 = st.columns(4)
    lr = h1.number_input("Learning rate η", min_value=0.001, max_value=2.0, value=0.10, step=0.01)
    max_ep = h2.slider("Training Epochs", 10, 500, 120)
    delay = h3.slider("Animation Delay (s)", 0.0, 0.6, 0.08, 0.02)
    stop_loss = h4.number_input("Early stop loss", min_value=0.0, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")

    a1, a2, a3 = st.columns(3)
    hidden_act = a1.selectbox("Hidden activation", list(ACTS.keys()), index=0)
    h_acts = [hidden_act] * n_hid
    o_act = a2.selectbox("Output activation", list(ACTS.keys()), index=1)
    loss_fn = a3.selectbox("Loss function", list(LOSSES.keys()), index=0)

    run_btn = st.button("🚀 Start Live Backprop Training", type="primary", use_container_width=True)

    if run_btn:
        if not input_ready:
            st.warning("Process a valid input row first.")
            render_chatbot(
        "backward propagation and gradient descent",
        system_prompt=(
            "You are a precise mathematician who loves the chain rule. "
            "You explain backpropagation with concrete numerical examples, step-by-step derivations, "
            "and clear notation. You make calculus feel approachable without dumbing it down."
        ),
        greeting=(
            "⬅️ Gradient Guide here. Backpropagation is just the chain rule applied systematically. "
            "Ask me about gradients, the chain rule, weight updates, vanishing gradients, or how to read the heatmaps."
        ),
        theme=MODULE_THEMES["backward_prop"],
        tutor_label="GRADIENT GUIDE ⬅️",
        placeholder="Ask about gradients, chain rule, or weight updates...",
    )
            return
        np.random.seed(int(seed))
        weights = make_weights(n_in, hidden_sizes)
        dashboard = st.empty()
        log_holder = st.empty()

        losses = []
        predictions = []
        grad_mags = []
        layer_grad_history = []
        layer_update_history = []
        epoch_records = []
        final_grads = None
        final_Zs = None
        final_As = None
        final_before = None
        final_after = None

        for ep in range(1, max_ep + 1):
            weights_before = [(W.copy(), b.copy()) for W, b in weights]
            Zs, As = forward_pass(X, weights, h_acts, o_act)
            y_pred = float(As[-1][0, 0])
            loss = float(LOSSES[loss_fn]["fn"](As[-1], np.array([[y_true]], dtype=float)))

            grads = backward_pass(weights, As, y_true, h_acts, o_act, loss_fn)
            layer_grad_vector = [float(np.mean(np.abs(g["dLdW"]))) for g in grads]
            layer_update_vector = [float(np.mean(np.abs(lr * g["dLdW"]))) for g in grads]
            mean_grad = float(np.mean(layer_grad_vector))

            updated_weights = []
            for (W, b), grad in zip(weights, grads):
                updated_weights.append((W - lr * grad["dLdW"], b - lr * grad["dLdb"]))
            weights = updated_weights

            losses.append(loss)
            predictions.append(y_pred)
            grad_mags.append(mean_grad)
            layer_grad_history.append(layer_grad_vector)
            layer_update_history.append(layer_update_vector)

            strongest_idx = int(np.argmax(layer_grad_vector))
            strongest_layer = labels[strongest_idx + 1]
            epoch_records.append(
                {
                    "epoch": ep,
                    "loss": loss,
                    "prediction": y_pred,
                    "mean_grad": mean_grad,
                    "strongest_layer": strongest_layer,
                }
            )

            live_script = _epoch_script(ep, y_pred, y_true, loss, mean_grad, strongest_layer)
            diag_vals = [Xv] + [As[i + 1].flatten().tolist() for i in range(len(sizes) - 2)] + [[y_pred]]

            with dashboard.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Epoch", f"{ep}/{max_ep}")
                m2.metric("Loss", f"{loss:.6f}")
                m3.metric("Prediction ŷ", f"{y_pred:.6f}")
                m4.metric("Avg Gradient", f"{mean_grad:.6f}")

                render_step_grid(
                    [
                        {
                            "eyebrow": "Output Error",
                            "title": "Prediction Gap",
                            "value": f"{y_pred - y_true:+.4f}",
                            "caption": "This is the signed difference between prediction and target.",
                            "accent": R,
                        },
                        {
                            "eyebrow": "Chain Rule",
                            "title": "Strongest Layer",
                            "value": strongest_layer,
                            "caption": "This layer currently receives the strongest average gradient signal.",
                            "accent": C,
                        },
                        {
                            "eyebrow": "Gradient Flow",
                            "title": "mean|dL/dW|",
                            "value": f"{mean_grad:.5f}",
                            "caption": "This measures the average strength of the learning update.",
                            "accent": G,
                        },
                        {
                            "eyebrow": "Optimizer Step",
                            "title": "η × grad",
                            "value": f"{lr * mean_grad:.5f}",
                            "caption": "Learning rate scales how large the actual weight step becomes.",
                            "accent": A,
                        },
                    ],
                    columns=4,
                )

                render_story_card(
                    "How this epoch pushed learning backward",
                    live_script,
                    eyebrow="Live Backprop Narration",
                    accent=C,
                    key_suffix=f"bp_epoch_{ep}",
                )

                c1, c2 = st.columns(2)
                if len(sizes) <= MAX_LAYERS:
                    c1.plotly_chart(
                        draw_network(sizes, labels, vals=diag_vals, highlight=strongest_idx + 1),
                        use_container_width=True,
                        key=f"bp_net_{ep}",
                    )
                c2.plotly_chart(
                    _gradient_chain_heatmap(grads, labels[1:], lr),
                    use_container_width=True,
                    key=f"bp_chain_heatmap_{ep}",
                )

                c3, c4 = st.columns(2)
                c3.plotly_chart(
                    line_story_chart(
                        [
                            {"name": "Loss", "x": list(range(1, ep + 1)), "y": losses, "color": R},
                            {"name": "Avg Gradient", "x": list(range(1, ep + 1)), "y": grad_mags, "color": A},
                        ],
                        "Loss And Gradient Story",
                        "Value",
                        height=320,
                    ),
                    use_container_width=True,
                    key=f"bp_story_{ep}",
                )
                c4.dataframe(_layer_update_table(grads, labels[1:], lr).round(5), use_container_width=True, hide_index=True)

            render_log(
                log_holder,
                [
                    f"Epoch {row['epoch']} | loss={row['loss']:.5f} | pred={row['prediction']:.5f} | avg_grad={row['mean_grad']:.5f} | strongest={row['strongest_layer']}"
                    for row in epoch_records[-8:]
                ],
            )

            final_grads = grads
            final_Zs = Zs
            final_As = As
            final_before = weights_before
            final_after = [(W.copy(), b.copy()) for W, b in weights]

            if delay > 0:
                time.sleep(delay)
            if loss <= stop_loss:
                break

        st.session_state.bp_result = {
            "sizes": sizes,
            "labels": labels,
            "arch_text": arch_text,
            "Xv": Xv,
            "y_true": float(y_true),
            "weights": weights,
            "final_Zs": final_Zs,
            "final_As": final_As,
            "final_grads": final_grads,
            "weights_before_final": final_before,
            "weights_after_final": final_after,
            "losses": losses,
            "predictions": predictions,
            "grad_mags": grad_mags,
            "layer_grad_history": layer_grad_history,
            "layer_update_history": layer_update_history,
            "loss": float(losses[-1]),
            "y_pred": float(predictions[-1]),
            "mean_grad": float(grad_mags[-1]),
            "epochs_run": ep,
            "lr": float(lr),
            "max_ep": int(max_ep),
            "loss_fn": loss_fn,
            "h_acts": h_acts,
            "o_act": o_act,
            "ai_attempted": False,
            "ai_pushed": False,
        }

    result = st.session_state.get("bp_result")
    if not result:
        render_chatbot(
        "backward propagation and gradient descent",
        system_prompt=(
            "You are a precise mathematician who loves the chain rule. "
            "You explain backpropagation with concrete numerical examples, step-by-step derivations, "
            "and clear notation. You make calculus feel approachable without dumbing it down."
        ),
        greeting=(
            "⬅️ Gradient Guide here. Backpropagation is just the chain rule applied systematically. "
            "Ask me about gradients, the chain rule, weight updates, vanishing gradients, or how to read the heatmaps."
        ),
        theme=MODULE_THEMES["backward_prop"],
        tutor_label="GRADIENT GUIDE ⬅️",
        placeholder="Ask about gradients, chain rule, or weight updates...",
    )
        return

    result = _prepare_ai_explanation(result)
    st.session_state.bp_result = result

    st.divider()
    section_header("4. AI Lesson + Audio", "A beginner-friendly story of how error became learning")
    render_story_card(
        "How this network turned error into updates",
        result["ai_text"],
        eyebrow=result["ai_label"],
        accent=C,
        key_suffix="bp_ai_story",
    )

    training_progress = min(100.0, max(0.0, (result["epochs_run"] / max(result["max_ep"], 1)) * 100.0))
    final_layer_names = result["labels"][1:]
    final_layer_strengths = [float(np.mean(np.abs(grad["dLdW"]))) for grad in result["final_grads"]]
    strongest_final_layer = final_layer_names[int(np.argmax(final_layer_strengths))] if final_layer_strengths else "Output"

    view_mode = render_visualization_mode("bp", accent=C, subject="backward propagation")

    render_summary_panel(
        "Backprop snapshot",
        [
            f"Architecture: `{result['arch_text']}`.",
            f"Training ran for `{result['epochs_run']}` epochs with learning rate `{result['lr']:.3f}`.",
            f"Final prediction is `{result['y_pred']:.6f}` for target `{result['y_true']:.6f}`, leaving a loss of `{result['loss']:.6f}`.",
            f"The strongest final learning signal is currently in `{strongest_final_layer}`.",
        ],
        eyebrow="WHAT TO NOTICE",
        accent=C,
    )

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(speedometer(min(result["loss"], 2.0), 2.0, "Loss", color=R, height=220), use_container_width=True, key="bp_g_loss")
    g2.plotly_chart(speedometer(min(result["mean_grad"], 1.0), 1.0, "Gradient", color=A, height=220), use_container_width=True, key="bp_g_grad")
    g3.plotly_chart(speedometer(training_progress, 100, "Epoch Progress", color=C, height=220), use_container_width=True, key="bp_g_prog")

    coach_narratives = {
        "Big Picture": (
            "Backpropagation turns a mistake into instructions. The loss says how wrong the output was, and gradients say how each parameter should move to reduce that error."
        ),
        "Chain Rule": (
            "The chain rule multiplies local sensitivities together. That is why a small activation derivative can weaken the learning signal before it reaches early layers."
        ),
        "Gradient Size": (
            "Large gradients create bold updates, while tiny gradients create slow learning. The best training often lives between chaos and stagnation."
        ),
        "How To Debug": (
            "Use the gradient matrix and delta matrix together. If gradients are healthy but updates are tiny, the learning rate is small. If both explode, the step size is too aggressive."
        ),
    }
    if view_mode == "Immersive Coach":
        render_ai_coach_panel("Coach focus", coach_narratives, key_suffix="bp_focus", accent=C)
    elif view_mode == "3D Visualization Explorer":
        st.plotly_chart(_gradient_3d_story(result), use_container_width=True, key="bp_3d_story_live")

    diag_vals = [result["Xv"]] + [result["final_As"][i + 1].flatten().tolist() for i in range(len(result["sizes"]) - 2)] + [[result["y_pred"]]]
    st.divider()
    section_header("5. Explore Learning Dynamics", "Begin with the training snapshot, then inspect one layer or open the deeper diagnostics")
    overview_tab, chain_tab, deep_tab = st.tabs(["Overview", "Chain Rule Explorer", "Deep Dive"])

    with overview_tab:
        render_step_grid(
            [
                {
                    "eyebrow": "Final Prediction",
                    "title": "ŷ",
                    "value": f"{result['y_pred']:.6f}",
                    "caption": "The output the network produced after training.",
                    "accent": C,
                },
                {
                    "eyebrow": "Target",
                    "title": "y",
                    "value": f"{result['y_true']:.6f}",
                    "caption": "The desired value the model was trying to match.",
                    "accent": G,
                },
                {
                    "eyebrow": "Loss",
                    "title": result["loss_fn"],
                    "value": f"{result['loss']:.6f}",
                    "caption": "This scalar summarizes how wrong the current prediction still is.",
                    "accent": R,
                },
                {
                    "eyebrow": "Gradient Strength",
                    "title": "mean|dL/dW|",
                    "value": f"{result['mean_grad']:.6f}",
                    "caption": "This tells you how aggressively the network is still learning.",
                    "accent": A,
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
                    key="bp_final_net",
                )
            else:
                st.info("The network is a little too deep to draw cleanly here, so use the gradient charts below to inspect how learning moved through the layers.")
        with c2:
            st.plotly_chart(
                line_story_chart(
                    [
                        {"name": "Loss", "x": list(range(1, len(result["losses"]) + 1)), "y": result["losses"], "color": R},
                        {"name": "Prediction", "x": list(range(1, len(result["predictions"]) + 1)), "y": result["predictions"], "color": C},
                    ],
                    "Prediction And Loss During Training",
                    "Value",
                    height=320,
                ),
                use_container_width=True,
                key="bp_pred_story_overview",
            )
        st.plotly_chart(
            _gradient_chain_heatmap(result["final_grads"], result["labels"][1:], result["lr"]),
            use_container_width=True,
            key="bp_chain_overview",
        )

    with chain_tab:
        layer_names = result["labels"][1:]
        layer_choice = st.selectbox("Inspect layer", layer_names, key="bp_focus_layer")
        layer_idx = layer_names.index(layer_choice)
        grad = result["final_grads"][layer_idx]

        render_step_grid(
            [
                {
                    "eyebrow": "Gradient",
                    "title": "|dL/dA|",
                    "value": f"{np.mean(np.abs(grad['dLdA'])):.6f}",
                    "caption": "How strongly the loss depends on this layer's activation output.",
                    "accent": C,
                },
                {
                    "eyebrow": "Activation Slope",
                    "title": "|dA/dZ|",
                    "value": f"{np.mean(np.abs(grad['dAdZ'])):.6f}",
                    "caption": "How much the activation function lets gradients pass through.",
                    "accent": G,
                },
                {
                    "eyebrow": "Layer Error",
                    "title": "|dL/dZ|",
                    "value": f"{np.mean(np.abs(grad['dLdZ'])):.6f}",
                    "caption": "The local error signal after the chain rule is applied.",
                    "accent": A,
                },
                {
                    "eyebrow": "Weight Update",
                    "title": "|eta·dL/dW|",
                    "value": f"{np.mean(np.abs(result['lr'] * grad['dLdW'])):.6f}",
                    "caption": "The average size of the actual weight change in this layer.",
                    "accent": R,
                },
            ],
            columns=4,
        )

        layer_script = (
            f"In {layer_choice}, the chain rule multiplies three ideas together: how much the loss cares about the layer output, "
            "how steep the activation function is, and how much each weight influenced the pre-activation score. "
            f"That produces the gradient matrix for {layer_choice}, which then gets scaled by the learning rate {result['lr']:.3f} to form the update."
        )
        render_story_card(
            f"How {layer_choice} receives its learning signal",
            layer_script,
            eyebrow="Layer Explanation",
            accent=C,
            key_suffix=f"bp_layer_audio_{layer_idx}",
        )

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            _gradient_matrix_heatmap(grad, f"{layer_choice} Gradient Matrix dL/dW"),
            use_container_width=True,
            key="bp_grad_matrix",
        )
        c2.plotly_chart(
            _weight_delta_heatmap(
                result["weights_before_final"][layer_idx],
                result["weights_after_final"][layer_idx],
                f"{layer_choice} Weight Change ΔW",
            ),
            use_container_width=True,
            key="bp_delta_matrix",
        )

    with deep_tab:
        story_tab, output_tab, weights_tab = st.tabs(["Gradient Story", "Output Story", "Final Weights"])

        with story_tab:
            layer_names = result["labels"][1:]
            st.plotly_chart(
                _layer_gradient_story(layer_names, result["layer_grad_history"], "Gradient Magnitude Per Layer Across Epochs", [C, G, A, R, "#8B5CF6"]),
                use_container_width=True,
                key="bp_grad_story",
            )
            st.plotly_chart(
                _layer_gradient_story(layer_names, result["layer_update_history"], "Update Magnitude Per Layer Across Epochs", [R, A, G, C, "#8B5CF6"]),
                use_container_width=True,
                key="bp_update_story",
            )

        with output_tab:
            c1, c2 = st.columns(2)
            c1.plotly_chart(
                line_story_chart(
                    [
                        {"name": "Loss", "x": list(range(1, len(result["losses"]) + 1)), "y": result["losses"], "color": R},
                        {"name": "Prediction", "x": list(range(1, len(result["predictions"]) + 1)), "y": result["predictions"], "color": C},
                    ],
                    "Prediction And Loss During Training",
                    "Value",
                    height=320,
                ),
                use_container_width=True,
                key="bp_pred_story",
            )
            c2.plotly_chart(
                contribution_bar(
                    ["Prediction - Target"],
                    [result["y_pred"] - result["y_true"]],
                    "Final Prediction Error",
                    positive=G,
                    negative=R,
                    neutral=A,
                    y_title="Signed error",
                    height=320,
                ),
                use_container_width=True,
                key="bp_final_error",
            )

        with weights_tab:
            st.caption("Open each table only when you want the raw trained parameters.")
            for idx, (W, b) in enumerate(result["weights"]):
                layer_name = "Output Layer" if idx == len(result["weights"]) - 1 else f"Hidden Layer {idx + 1}"
                with st.expander(layer_name, expanded=False):
                    df = pd.DataFrame(W)
                    df["Bias"] = b.flatten()
                    st.dataframe(df.round(5), use_container_width=True)

    render_chatbot(
        "backward propagation and gradient descent",
        system_prompt=(
            "You are a precise mathematician who loves the chain rule. "
            "You explain backpropagation with concrete numerical examples, step-by-step derivations, "
            "and clear notation. You make calculus feel approachable without dumbing it down."
        ),
        greeting=(
            "⬅️ Gradient Guide here. Backpropagation is just the chain rule applied systematically. "
            "Ask me about gradients, the chain rule, weight updates, vanishing gradients, or how to read the heatmaps."
        ),
        theme=MODULE_THEMES["backward_prop"],
        tutor_label="GRADIENT GUIDE ⬅️",
        placeholder="Ask about gradients, chain rule, or weight updates...",
    )
if __name__ == "__main__":
    backward_propagation_page()
