import base64
import hashlib
import io
import json
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from PIL import Image, ImageFilter

from utils.ai_helper import OPENAI_OK, get_ai_explanation, get_api_key
from utils.chatbot import push_tutor_insight, render_chatbot
from utils.learning_ui import (
    contribution_bar,
    heatmap_with_text,
    line_story_chart,
    render_ai_coach_panel,
    render_learning_journey,
    render_step_grid,
    render_visualization_mode,
    scatter3d_story,
)
from utils.nn_helpers import A, C, G, R, plotly_layout
from utils.styles import gradient_header, inject_global_css, render_log, section_header, speedometer
from utils.voice import render_voice_button

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False


GRID_SIDE = 16
N = GRID_SIDE * GRID_SIDE


class HopfieldEngine:
    def __init__(self, size=N):
        self.N = size
        self.W = np.zeros((size, size))

    def store(self, pattern):
        vec = pattern.reshape(-1, 1)
        self.W = (vec @ vec.T) / self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        return -0.5 * float(state @ self.W @ state)

    def recover(self, state, steps=120):
        curr = state.copy()
        energies = [self.energy(curr)]
        for _ in range(steps):
            idx = np.random.randint(0, self.N)
            curr[idx] = 1.0 if (self.W[idx] @ curr) >= 0 else -1.0
            energies.append(self.energy(curr))
            stable = np.array_equal(curr, np.where(self.W @ curr >= 0, 1.0, -1.0))
            if stable:
                break
        return curr, energies


def _resample():
    return Image.Resampling.LANCZOS


def _clean_canvas_image(data, out_size=192):
    if data is None or not isinstance(data, np.ndarray):
        return None
    rgba = Image.fromarray(data.astype("uint8"), "RGBA")
    gray = rgba.convert("L")
    arr = np.array(gray)
    bw = np.where(arr > 50, 0, 255).astype("uint8")
    filtered = Image.fromarray(bw, "L").filter(ImageFilter.BoxBlur(1))
    final = np.where(np.array(filtered) < 220, 0, 255).astype("uint8")
    return Image.fromarray(final, "L").resize((out_size, out_size), _resample())


def _image_to_bipolar(img):
    small = img.resize((GRID_SIDE, GRID_SIDE), _resample())
    arr = np.array(small)
    return np.where(arr < 200, 1.0, -1.0).flatten()


def _canvas_hash(data):
    if data is None or not isinstance(data, np.ndarray):
        return None
    return hashlib.md5(data.tobytes()).hexdigest()


def _is_blank(data):
    img = _clean_canvas_image(data, out_size=64)
    if img is None:
        return True
    arr = np.array(img)
    return int(np.sum(arr < 180)) < 18


def _image_to_base64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _plot_grid(vec, title):
    arr = np.array(vec).reshape(GRID_SIDE, GRID_SIDE)
    return heatmap_with_text(
        arr,
        [str(i + 1) for i in range(GRID_SIDE)],
        [str(i + 1) for i in range(GRID_SIDE)],
        title,
        zmid=0,
        height=320,
        colorbar_title="Neuron state",
    )


def _plot_energy(energies):
    return line_story_chart(
        [{"name": "Energy", "x": list(range(len(energies))), "y": energies, "color": "#8B5CF6"}],
        "Hopfield Energy Descent",
        "Energy",
        height=300,
    )


def _plot_weight_surface(weight_matrix):
    reduced = weight_matrix[::4, ::4]
    fig = go.Figure(
        go.Surface(
            z=reduced,
            colorscale="Turbo",
            colorbar=dict(title="Weight"),
        )
    )
    fig.update_layout(
        title=dict(text="3D Synaptic Landscape", font=dict(color="#FFFFFF", family="Montserrat", size=18)),
        scene=dict(
            xaxis=dict(title="Neuron x", color="#94A3B8"),
            yaxis=dict(title="Neuron y", color="#94A3B8"),
            zaxis=dict(title="Weight", color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        **plotly_layout(height=420, margin=dict(l=0, r=0, t=55, b=0)),
    )
    return fig


def _plot_state_3d(input_vec, recovered_vec):
    coords_in = np.argwhere(np.array(input_vec).reshape(GRID_SIDE, GRID_SIDE) > 0)
    coords_out = np.argwhere(np.array(recovered_vec).reshape(GRID_SIDE, GRID_SIDE) > 0)
    return scatter3d_story(
        [
            {
                "name": "Input strokes",
                "x": coords_in[:, 1].tolist() if len(coords_in) else [],
                "y": coords_in[:, 0].tolist() if len(coords_in) else [],
                "z": [1.0] * len(coords_in),
                "mode": "markers",
                "size": 6,
                "color": C,
                "line_color": C,
                "text": [f"Input neuron ({y}, {x})" for y, x in coords_in],
            },
            {
                "name": "Recovered state",
                "x": coords_out[:, 1].tolist() if len(coords_out) else [],
                "y": coords_out[:, 0].tolist() if len(coords_out) else [],
                "z": [2.0] * len(coords_out),
                "mode": "markers",
                "size": 6,
                "color": G,
                "line_color": G,
                "text": [f"Recovered neuron ({y}, {x})" for y, x in coords_out],
            },
        ],
        "3D Drawing And Recovery View",
        "x",
        "y",
        "State plane",
        height=430,
    )


def _parse_detection_response(text):
    fields = {"name": "Unknown", "category": "Sketch", "confidence": 70, "note": text.strip() or "No additional analysis."}
    patterns = {
        "name": r"NAME\s*:\s*(.+)",
        "category": r"CATEGORY\s*:\s*(.+)",
        "confidence": r"CONFIDENCE\s*:\s*(.+)",
        "note": r"NOTE\s*:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()

    confidence_match = re.search(r"(\d+)", str(fields["confidence"]))
    if confidence_match:
        fields["confidence"] = max(0, min(100, int(confidence_match.group(1))))
    else:
        fields["confidence"] = 70
    return fields


def _detect_with_ai(clean_img):
    key = get_api_key()
    if not key or not OPENAI_OK or OpenAI is None:
        return {
            "name": "Unknown",
            "category": "Sketch",
            "confidence": 0,
            "note": "Vision API is unavailable, so direct sketch detection could not run.",
            "raw": "Vision API unavailable.",
        }

    b64 = _image_to_base64(clean_img)
    prompt = (
        "Look at this single hand-drawn black sketch on a white background. "
        "Identify what was drawn as accurately as possible, even if it is a letter, digit, symbol, object, shape, or doodle. "
        "Reply in exactly four lines:\n"
        "NAME: <short label>\n"
        "CATEGORY: <letter, number, shape, symbol, object, doodle, or word>\n"
        "CONFIDENCE: <0-100>\n"
        "NOTE: <one short sentence>"
    )

    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=120,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_detection_response(raw)
        parsed["raw"] = raw
        return parsed
    except Exception as exc:
        return {
            "name": "Unknown",
            "category": "Sketch",
            "confidence": 0,
            "note": f"Vision API error: {exc}",
            "raw": f"Vision API error: {exc}",
        }


def _prepare_ai_analysis(result):
    if result.get("ai_attempted"):
        return result

    prompt = (
        "Explain this sketch-detection and Hopfield-matrix result to a complete beginner. "
        f"The drawing was identified as {result['detected_name']} in category {result['detected_category']} "
        f"with confidence {result['detected_confidence']}. "
        f"The Hopfield energy moved from {result['energies'][0]:.4f} to {result['energies'][-1]:.4f}. "
        "Explain what the matrix means, what the energy means, and how the sketch detector and Hopfield matrix views complement each other."
    )
    ai_text = get_ai_explanation(
        prompt,
        system_prompt=(
            "You are an immersive AI coach for neural networks. "
            "Explain in 4 to 6 short paragraphs using simple analogies and visual language for beginners."
        ),
        max_tokens=340,
    )
    fallback = (
        f"The drawing detector believes you sketched `{result['detected_name']}`. "
        f"The 16 by 16 Hopfield-style matrix is a compressed neural map of your strokes, and the falling energy curve shows the network settling into a stable internal state."
    )
    result["ai_text"] = ai_text or fallback
    result["ai_label"] = "AI Coach // Sketch Interpretation Guide" if ai_text else "Sketch Interpretation Guide"
    result["ai_attempted"] = True
    if not result.get("ai_pushed"):
        push_tutor_insight(result["ai_text"], result["ai_label"])
        result["ai_pushed"] = True
    return result


def _analyze_drawing(clean_img, analysis_hash):
    input_vec = _image_to_bipolar(clean_img)
    engine = HopfieldEngine(N)
    engine.store(input_vec)
    recovered, energies = engine.recover(input_vec, steps=90)
    changed = (recovered != input_vec).astype(float)
    detection = _detect_with_ai(clean_img)
    return {
        "analysis_hash": analysis_hash,
        "clean_img": clean_img,
        "input_vec": input_vec,
        "engine": engine,
        "recovered": recovered,
        "energies": energies,
        "changed": changed,
        "detected_name": detection["name"],
        "detected_category": detection["category"],
        "detected_confidence": detection["confidence"],
        "detected_note": detection["note"],
        "detected_raw": detection["raw"],
        "ai_attempted": False,
        "ai_pushed": False,
    }


def _init_state():
    if "hop_canvas_key" not in st.session_state:
        st.session_state.hop_canvas_key = 0
    if "hop_result" not in st.session_state:
        st.session_state.hop_result = None


def _clear_canvas():
    st.session_state.hop_canvas_key += 1
    st.session_state.hop_result = None


def main():
    inject_global_css()
    _init_state()

    gradient_header(
        "Hopfield Network",
        "Free-Form Sketch Detection + Live Matrix View · Detect what you draw and inspect the neural matrix instantly",
        "🧠",
    )

    theory_text = (
        "This page now reads your drawing directly and tells you what it looks like. "
        "Alongside that, it converts your sketch into a Hopfield-style bipolar matrix so you can see how the drawing becomes a neural state."
    )
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown(
            """
            **Free-Form Sketch Detection + Hopfield Matrix**

            1. Clean the canvas into a sharp black-on-white sketch image
            2. Ask a vision model what the drawing most likely represents
            3. Downsample the drawing into a 16 × 16 bipolar neuron matrix
            4. Build a Hopfield-style weight matrix from the current sketch and observe its energy behavior
            """
        )
        render_voice_button(theory_text, key_suffix="hop_theory")

    render_learning_journey(
        "Draw Anything And Get A Direct Output",
        "This version is no longer limited to stored memory patterns. It detects the free-form sketch directly, then shows you the neural matrix and energy views built from the exact drawing you made.",
        [
            "The visible label comes from direct sketch analysis, not from matching against a tiny fixed memory bank.",
            "The 16 by 16 matrix is a compact neural representation of your strokes.",
            "The Hopfield weight matrix now reflects the current sketch itself, so the matrix view always matches what you drew.",
            "Analysis runs on demand when you click the button, which keeps the page much faster.",
        ],
        "Think of the system as having two eyes. One eye describes what the drawing looks like, and the other eye turns the same drawing into a grid of neural states and energy patterns.",
        audio_text=theory_text,
        key_suffix="hop_intro",
    )

    section_header("1. Draw And Analyze", "Sketch freely, then run detection only when you want the result")
    top1, top2 = st.columns([1.2, 1])
    with top1:
        view_mode = render_visualization_mode("hop", accent=C, subject="the sketch detector and Hopfield matrix lab")
    with top2:
        explain_with_ai = st.checkbox("Generate AI coach explanation", value=False, key="hop_explain_ai")
        st.caption("Keeping this off makes the page faster. You can still analyze the drawing output immediately.")

    if CANVAS_OK:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#00f0ff",
            background_color="#0f172a",
            height=360,
            width=900,
            drawing_mode="freedraw",
            key=f"hop_canvas_{st.session_state.hop_canvas_key}",
        )
    else:
        st.error("Missing `streamlit-drawable-canvas`, so the drawing board is unavailable.")
        render_chatbot("free-form sketch detection and Hopfield matrix visualization")
        return

    action_cols = st.columns([1, 1, 4])
    analyze_clicked = action_cols[0].button("Analyze Drawing", use_container_width=True, type="primary")
    clear_clicked = action_cols[1].button("Clear", use_container_width=True)
    if clear_clicked:
        _clear_canvas()
        st.rerun()

    current_canvas_hash = _canvas_hash(canvas.image_data) if canvas.image_data is not None else None
    existing_result = st.session_state.get("hop_result")
    if existing_result and current_canvas_hash and existing_result.get("analysis_hash") != current_canvas_hash:
        st.info("The drawing changed after the last analysis. Click `Analyze Drawing` to refresh the output.")

    if analyze_clicked:
        if canvas.image_data is None or _is_blank(canvas.image_data):
            st.warning("Draw something first, then click Analyze Drawing.")
        else:
            clean_img = _clean_canvas_image(canvas.image_data)
            analysis_hash = _canvas_hash(canvas.image_data)
            cached = st.session_state.get("hop_result")
            if not cached or cached.get("analysis_hash") != analysis_hash:
                with st.spinner("Analyzing your sketch..."):
                    result = _analyze_drawing(clean_img, analysis_hash)
                    st.session_state.hop_result = result
            if explain_with_ai and st.session_state.hop_result:
                with st.spinner("Generating AI coach explanation..."):
                    st.session_state.hop_result = _prepare_ai_analysis(st.session_state.hop_result)

    result = st.session_state.get("hop_result")
    if not result:
        render_chatbot("free-form sketch detection and Hopfield matrix visualization")
        return

    energy_drop = float(result["energies"][0] - result["energies"][-1]) if len(result["energies"]) > 1 else 0.0
    stability = float(np.mean(result["recovered"] == result["input_vec"]) * 100.0)

    section_header("2. Detection Output", "What the system thinks you drew and how stable the matrix became")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(speedometer(result["detected_confidence"], 100, "Detection Confidence", color=G, height=220), use_container_width=True, key="hop_g_conf")
    g2.plotly_chart(speedometer(min(abs(energy_drop), 100), 100, "Energy Drop", color=A, height=220), use_container_width=True, key="hop_g_energy")
    g3.plotly_chart(speedometer(stability, 100, "State Stability", color=C, height=220), use_container_width=True, key="hop_g_stability")

    render_step_grid(
        [
            {
                "eyebrow": "Detected Drawing",
                "title": result["detected_name"],
                "value": f"{result['detected_confidence']}%",
                "caption": f"Category: {result['detected_category']}",
                "accent": G,
            },
            {
                "eyebrow": "Matrix Size",
                "title": "Neuron Grid",
                "value": f"{GRID_SIDE} x {GRID_SIDE}",
                "caption": "This is the compressed neural matrix created from your exact sketch.",
                "accent": C,
            },
            {
                "eyebrow": "Energy",
                "title": "Drop",
                "value": f"{energy_drop:+.3f}",
                "caption": "This shows how the Hopfield-style state settles into a stable configuration.",
                "accent": A,
            },
            {
                "eyebrow": "Interpretation",
                "title": "Note",
                "value": result["detected_category"],
                "caption": result["detected_note"],
                "accent": R,
            },
        ],
        columns=4,
    )

    coach_narratives = {
        "Big Picture": (
            f"The detector sees your sketch as `{result['detected_name']}`. "
            "The matrix views underneath are not a separate memory lookup; they are built directly from the exact drawing you just made."
        ),
        "How The Matrix Works": (
            "Every cell in the 16 by 16 grid acts like a neuron that is either active or inactive depending on whether that part of the drawing contains stroke information."
        ),
        "What The Energy Means": (
            "The energy plot shows how self-consistent the sketch state is under the Hopfield-style weight matrix created from the same drawing."
        ),
        "Why This Is Faster": (
            "The app now waits for your Analyze button before calling detection or building the heavy charts. That avoids expensive recomputation on every tiny brush movement."
        ),
    }
    if view_mode == "Immersive Coach":
        render_ai_coach_panel("Coach focus", coach_narratives, key_suffix="hop_focus", accent=C)

    with st.container(border=True):
        st.caption("DETECTION RESULT")
        st.markdown(f"### {result['detected_name']}")
        st.write(result["detected_note"])
        with st.expander("Raw detector response", expanded=False):
            st.code(result["detected_raw"])

    if explain_with_ai:
        if not result.get("ai_attempted"):
            result = _prepare_ai_analysis(result)
            st.session_state.hop_result = result
        with st.container(border=True):
            st.caption(result["ai_label"].upper())
            st.write(result["ai_text"])
        render_voice_button(result["ai_text"], key_suffix="hop_ai_story")

    if view_mode == "3D Visualization Explorer":
        section_header("3D Explorer", "Interactive 3D views for the exact sketch you analyzed")
        d1, d2 = st.columns(2)
        d1.plotly_chart(_plot_state_3d(result["input_vec"], result["recovered"]), use_container_width=True, key="hop_3d_state_live")
        d2.plotly_chart(_plot_weight_surface(result["engine"].W), use_container_width=True, key="hop_weight_surface_live")

    log_lines = [
        f"Detected drawing: {result['detected_name']}",
        f"Category: {result['detected_category']}",
        f"Detection confidence: {result['detected_confidence']}%",
        f"Energy start -> end: {result['energies'][0]:.4f} -> {result['energies'][-1]:.4f}",
        f"State stability: {stability:.2f}%",
    ]
    render_log(st.empty(), log_lines)

    section_header("3. Matrix Views", "See the sketch, the neural matrix, and the Hopfield-style weight structure from the same input")
    tabs = st.tabs(["🧠 Matrix View", "📈 Energy + Scores", "⚙ Weight Matrix", "🖼 Sketch Preview"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(_plot_grid(result["input_vec"], "Input Matrix"), use_container_width=True, key="hop_input")
        c2.plotly_chart(_plot_grid(result["recovered"], "Recovered Matrix"), use_container_width=True, key="hop_recovered")
        c3.plotly_chart(_plot_grid(np.where(result["changed"] > 0, 1.0, -1.0), "Changed Cells"), use_container_width=True, key="hop_changed")

    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(_plot_energy(result["energies"]), use_container_width=True, key="hop_energy")
        c2.plotly_chart(
            contribution_bar(
                ["Detection Confidence", "State Stability", "Energy Drop"],
                [float(result["detected_confidence"]), stability, float(min(abs(energy_drop), 100.0))],
                "Analysis Summary",
                positive=G,
                negative=R,
                neutral=A,
                height=320,
                y_title="Score",
            ),
            use_container_width=True,
            key="hop_summary_scores",
        )

    with tabs[2]:
        st.plotly_chart(
            heatmap_with_text(
                result["engine"].W[::4, ::4],
                [f"n{i+1}" for i in range(result["engine"].W[::4, ::4].shape[1])],
                [f"n{i+1}" for i in range(result["engine"].W[::4, ::4].shape[0])],
                "Downsampled Weight Matrix Built From Your Drawing",
                zmid=0,
                height=360,
                colorbar_title="Weight",
            ),
            use_container_width=True,
            key="hop_weights",
        )
        if view_mode == "3D Visualization Explorer":
            st.plotly_chart(_plot_weight_surface(result["engine"].W), use_container_width=True, key="hop_weight_surface")

    with tabs[3]:
        st.image(result["clean_img"], caption="Cleaned sketch used for direct detection and matrix conversion", use_container_width=True)

    render_chatbot("free-form sketch detection and Hopfield matrix visualization")


if __name__ == "__main__":
    main()
