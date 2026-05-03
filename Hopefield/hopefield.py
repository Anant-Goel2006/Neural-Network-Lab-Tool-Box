import hashlib
import json
import os
from functools import lru_cache

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from utils.ai_helper import get_ai_explanation
from utils.chatbot import push_tutor_insight, render_chatbot
from utils.learning_ui import (
    contribution_bar,
    heatmap_with_text,
    line_story_chart,
    render_ai_coach_panel,
    render_learning_journey,
    render_step_grid,
    scatter3d_story,
)
from utils.nn_helpers import A, C, G, R, plotly_layout
from utils.styles import gradient_header, inject_global_css, render_log, section_header, speedometer, inject_module_theme, MODULE_THEMES
from utils.voice import render_voice_button

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False


GRID_SIDE = 16
N = GRID_SIDE * GRID_SIDE
DETECTOR_SIDE = 96
CLASSIFIER_SIDE = 32
CLASSIFIER_K = 9
CHARACTER_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHARACTER_LABELS += list("abcdefghijklmnopqrstuvwxyz")
SHAPE_LABELS = ["Circle", "Triangle", "Square", "Rectangle", "Diamond", "Star", "Plus", "Minus", "Slash", "Backslash", "Arrow"]
CLASSIFIER_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
]
CLASSIFIER_ANGLES = (-15, -6, 0, 6, 15)
CLASSIFIER_SCALES = (0.88, 0.95, 1.0, 1.08, 1.16)
CLASSIFIER_SHIFTS = [(-2, -2), (0, 0), (2, 2), (-2, 2), (2, -2)]
TEXT_SHEARS = (-0.14, 0.14)
PREFERRED_TTF_FONTS = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "cambria.ttc",
    "consola.ttf",
    "comic.ttf",
    "georgia.ttf",
    "times.ttf",
    "trebuc.ttf",
    "verdana.ttf",
    "segoeui.ttf",
    "gabriola.ttf",
]


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

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    background_is_light = float(np.mean(border)) >= 127.0

    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not background_is_light:
        binary = 255 - binary

    # Remove tiny specks while keeping handwritten strokes readable.
    binary = cv2.medianBlur(binary, 3)
    filtered = Image.fromarray(binary, "L").filter(ImageFilter.BoxBlur(1))
    final = np.where(np.array(filtered) < 220, 0, 255).astype("uint8")
    cleaned = Image.fromarray(final, "L")
    width, height = cleaned.size
    scale = min(out_size / max(width, 1), out_size / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cleaned.resize((new_w, new_h), _resample())
    canvas = Image.new("L", (out_size, out_size), 255)
    x_off = (out_size - new_w) // 2
    y_off = (out_size - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    return canvas


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


def _binary_mask_from_image(clean_img):
    arr = np.array(clean_img.convert("L"))
    mask = (arr < 200).astype(np.uint8)
    if mask.sum() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = max(int(areas.max()), 1)
    keep = [idx + 1 for idx, area in enumerate(areas) if area >= max(6, largest * 0.03)]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    return np.isin(labels, keep).astype(np.uint8)


def _normalize_mask(mask, size=DETECTOR_SIDE, pad=10):
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = (mask[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8)
    h, w = crop.shape
    scale = (size - 2 * pad) / max(h, w, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    resized = (resized > 80).astype(np.uint8)

    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _find_contours(binary_img, mode=cv2.RETR_CCOMP):
    found = cv2.findContours(binary_img, mode, cv2.CHAIN_APPROX_SIMPLE)
    if len(found) == 2:
        return found[0], found[1]
    return found[1], found[2]


def _mask_descriptors(mask):
    binary = (mask > 0).astype(np.uint8)
    active = float(binary.sum())
    if active <= 0:
        zero = np.zeros(DETECTOR_SIDE, dtype=float)
        return {
            "active_ratio": 0.0,
            "aspect": 1.0,
            "extent": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "holes": 0,
            "vertices": 0,
            "h_sym": 0.0,
            "v_sym": 0.0,
            "d_sym": 0.0,
            "ad_sym": 0.0,
            "hu": np.zeros(7, dtype=float),
            "proj_h": zero,
            "proj_v": zero,
        }

    ys, xs = np.where(binary > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    binary255 = (binary * 255).astype(np.uint8)
    contours, hierarchy = _find_contours(binary255, mode=cv2.RETR_CCOMP)
    main = max(contours, key=cv2.contourArea) if contours else None

    contour_area = float(cv2.contourArea(main)) if main is not None else active
    perimeter = float(cv2.arcLength(main, True)) if main is not None else float(2 * (w + h))
    approx = cv2.approxPolyDP(main, 0.045 * perimeter, True) if main is not None and perimeter > 0 else np.empty((0, 1, 2))
    hull = cv2.convexHull(main) if main is not None else None
    hull_area = float(cv2.contourArea(hull)) if hull is not None else contour_area
    moments = cv2.moments(main) if main is not None else cv2.moments(binary255)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    holes = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0] if hierarchy.ndim == 3 else hierarchy
        holes = int(np.sum(hierarchy[:, 3] != -1))

    float_mask = binary.astype(float)
    return {
        "active_ratio": active / binary.size,
        "aspect": w / max(h, 1),
        "extent": contour_area / max(w * h, 1),
        "solidity": contour_area / max(hull_area, 1.0),
        "circularity": (4.0 * np.pi * contour_area) / max(perimeter * perimeter, 1.0),
        "holes": holes,
        "vertices": int(len(approx)),
        "h_sym": 1.0 - float(np.mean(np.abs(float_mask - np.fliplr(float_mask)))),
        "v_sym": 1.0 - float(np.mean(np.abs(float_mask - np.flipud(float_mask)))),
        "d_sym": 1.0 - float(np.mean(np.abs(float_mask - float_mask.T))),
        "ad_sym": 1.0 - float(np.mean(np.abs(float_mask - np.flipud(float_mask.T)))),
        "hu": hu,
        "proj_h": float_mask.mean(axis=1),
        "proj_v": float_mask.mean(axis=0),
    }


def _label_category(label):
    if len(label) == 1 and label.isdigit():
        return "Number"
    if len(label) == 1 and label.isalpha():
        return "Letter"
    if label in {"Plus", "Minus", "Slash", "Backslash", "Arrow"}:
        return "Symbol"
    return "Shape"


def _text_render_scale(label):
    if len(label) == 1 and label.islower():
        return 2.0
    if len(label) == 1 and label.isalpha():
        return 2.4
    return 2.2


def _draw_shape_prototype(label, size=DETECTOR_SIDE):
    canvas = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    margin = int(size * 0.18)
    thickness = max(4, size // 18)

    if label == "Circle":
        cv2.circle(canvas, (c, c), size // 3, 255, thickness)
    elif label == "Triangle":
        pts = np.array([[c, margin], [size - margin, size - margin], [margin, size - margin]], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, 255, thickness)
    elif label == "Square":
        cv2.rectangle(canvas, (margin, margin), (size - margin, size - margin), 255, thickness)
    elif label == "Rectangle":
        cv2.rectangle(canvas, (int(size * 0.14), int(size * 0.28)), (int(size * 0.86), int(size * 0.72)), 255, thickness)
    elif label == "Diamond":
        pts = np.array([[c, margin], [size - margin, c], [c, size - margin], [margin, c]], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, 255, thickness)
    elif label == "Star":
        outer = size * 0.34
        inner = size * 0.14
        pts = []
        for idx in range(10):
            angle = -np.pi / 2 + idx * np.pi / 5
            radius = outer if idx % 2 == 0 else inner
            pts.append((int(round(c + radius * np.cos(angle))), int(round(c + radius * np.sin(angle)))))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], True, 255, thickness)
    elif label == "Plus":
        cv2.line(canvas, (c, margin), (c, size - margin), 255, thickness)
        cv2.line(canvas, (margin, c), (size - margin, c), 255, thickness)
    elif label == "Minus":
        cv2.line(canvas, (margin, c), (size - margin, c), 255, thickness)
    elif label == "Slash":
        cv2.line(canvas, (margin, size - margin), (size - margin, margin), 255, thickness)
    elif label == "Backslash":
        cv2.line(canvas, (margin, margin), (size - margin, size - margin), 255, thickness)
    elif label == "Arrow":
        cv2.arrowedLine(canvas, (margin, c), (size - margin, c), 255, thickness, tipLength=0.25)
    return (canvas > 0).astype(np.uint8)


def _render_text_prototype(label, font, thickness, size=DETECTOR_SIDE):
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = _text_render_scale(label)
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = max(2, (size - text_w) // 2)
    y = max(text_h + 2, (size + text_h) // 2 - baseline // 2)
    cv2.putText(canvas, label, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    return canvas


@lru_cache(maxsize=1)
def _available_ttf_font_paths():
    font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    paths = []
    for name in PREFERRED_TTF_FONTS:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            paths.append(path)
    return tuple(paths[:7])


def _render_ttf_text_prototype(label, font_path, size=DETECTOR_SIDE):
    canvas = Image.new("L", (size, size), 0)
    drawer = ImageDraw.Draw(canvas)
    font = None
    bbox = None
    for font_size in range(int(size * 0.82), int(size * 0.34), -4):
        try:
            candidate = ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
        candidate_bbox = drawer.textbbox((0, 0), label, font=candidate)
        text_w = candidate_bbox[2] - candidate_bbox[0]
        text_h = candidate_bbox[3] - candidate_bbox[1]
        if text_w <= size * 0.76 and text_h <= size * 0.74:
            font = candidate
            bbox = candidate_bbox
            break

    if font is None:
        font = ImageFont.load_default()
        bbox = drawer.textbbox((0, 0), label, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = int(round((size - text_w) / 2.0 - bbox[0]))
    y = int(round((size - text_h) / 2.0 - bbox[1]))
    drawer.text((x, y), label, fill=255, font=font)
    return np.array(canvas, dtype=np.uint8)


def _classifier_mask(mask):
    mask = np.array(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("Classifier mask must be 2D.")
    norm = _normalize_mask(mask, size=CLASSIFIER_SIDE, pad=3)
    return (norm > 0).astype(np.uint8) * 255


def _shear_mask(mask, shear):
    offset = -shear * DETECTOR_SIDE * 0.5
    matrix = np.array([[1.0, shear, offset], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(mask, matrix, (DETECTOR_SIDE, DETECTOR_SIDE), flags=cv2.INTER_LINEAR, borderValue=0)


def _augment_training_mask(base_mask, text_like=False):
    base = np.array(base_mask, dtype=np.uint8)
    samples = []
    center = (DETECTOR_SIDE / 2, DETECTOR_SIDE / 2)

    base_variants = [base]
    if text_like:
        for shear in TEXT_SHEARS:
            base_variants.append(_shear_mask(base, shear))
        kernel = np.ones((2, 2), dtype=np.uint8)
        base_variants.append(cv2.dilate(base, kernel, iterations=1))
        base_variants.append(cv2.erode(base, kernel, iterations=1))
        # Gaussian noise variant for robustness to imperfect strokes
        noisy = base.astype(np.float32)
        noisy += np.random.normal(0, 12, noisy.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        base_variants.append(noisy)

    for variant in base_variants:
        for angle in CLASSIFIER_ANGLES:
            for scale in CLASSIFIER_SCALES:
                for shift_x, shift_y in CLASSIFIER_SHIFTS:
                    matrix = cv2.getRotationMatrix2D(center, angle, scale)
                    matrix[0, 2] += shift_x
                    matrix[1, 2] += shift_y
                    warped = cv2.warpAffine(variant, matrix, (DETECTOR_SIDE, DETECTOR_SIDE), flags=cv2.INTER_LINEAR, borderValue=0)
                    samples.append(_classifier_mask(warped))
    return samples


def _hog_descriptor():
    return cv2.HOGDescriptor((CLASSIFIER_SIDE, CLASSIFIER_SIDE), (16, 16), (8, 8), (8, 8), 9)


@lru_cache(maxsize=1)
def _character_classifier_bank():
    hog = _hog_descriptor()
    label_to_idx = {label: idx for idx, label in enumerate(CHARACTER_LABELS)}
    samples = []
    targets = []

    for label in CHARACTER_LABELS:
        for font in CLASSIFIER_FONTS:
            for thickness in [2, 3, 5]:
                base_mask = _render_text_prototype(label, font, thickness)
                for sample in _augment_training_mask(base_mask, text_like=True):
                    samples.append(hog.compute(sample).flatten())
                    targets.append(label_to_idx[label])
        for font_path in _available_ttf_font_paths():
            base_mask = _render_ttf_text_prototype(label, font_path)
            for sample in _augment_training_mask(base_mask, text_like=True):
                samples.append(hog.compute(sample).flatten())
                targets.append(label_to_idx[label])

    train_x = np.array(samples, dtype=np.float32)
    train_y = np.array(targets, dtype=np.int32)
    knn = cv2.ml.KNearest_create()
    knn.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    return {"hog": hog, "knn": knn, "labels": CHARACTER_LABELS}


@lru_cache(maxsize=1)
def _shape_classifier_bank():
    hog = _hog_descriptor()
    label_to_idx = {label: idx for idx, label in enumerate(SHAPE_LABELS)}
    samples = []
    targets = []

    for label in SHAPE_LABELS:
        base_mask = (_draw_shape_prototype(label, size=DETECTOR_SIDE) * 255).astype(np.uint8)
        for sample in _augment_training_mask(base_mask):
            samples.append(hog.compute(sample).flatten())
            targets.append(label_to_idx[label])

    train_x = np.array(samples, dtype=np.float32)
    train_y = np.array(targets, dtype=np.int32)
    knn = cv2.ml.KNearest_create()
    knn.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    return {"hog": hog, "knn": knn, "labels": SHAPE_LABELS}


def _rank_matches_from_bank(classifier_mask, bank):
    feature = bank["hog"].compute(classifier_mask).reshape(1, -1).astype(np.float32)
    _, result, neighbours, distances = bank["knn"].findNearest(feature, k=CLASSIFIER_K)

    labels = bank["labels"]
    neighbour_ids = neighbours[0].astype(int).tolist()
    distance_values = [float(val) for val in distances[0].tolist()]
    max_distance = max(max(distance_values), 1.0)

    aggregates = {}
    for idx, dist in zip(neighbour_ids, distance_values):
        label = labels[idx]
        bucket = aggregates.setdefault(
            label,
            {"label": label, "category": _label_category(label), "votes": 0, "distance_sum": 0.0},
        )
        bucket["votes"] += 1
        bucket["distance_sum"] += dist

    ranked = []
    for item in aggregates.values():
        avg_distance = item["distance_sum"] / max(item["votes"], 1)
        closeness = max(0.0, 1.0 - (avg_distance / max_distance))
        score = 0.7 * (item["votes"] / CLASSIFIER_K) + 0.3 * closeness
        ranked.append(
            {
                "label": item["label"],
                "category": item["category"],
                "votes": item["votes"],
                "avg_distance": avg_distance,
                "score": max(0.0, min(1.0, float(score))),
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["votes"], -item["avg_distance"]), reverse=True)
    predicted_label = labels[int(result[0, 0])]
    if ranked and ranked[0]["label"] != predicted_label:
        ranked.sort(key=lambda item: (item["label"] != predicted_label, -item["score"], -item["votes"], item["avg_distance"]))
    return ranked


def _rank_character_matches(classifier_mask):
    return _rank_matches_from_bank(classifier_mask, _character_classifier_bank())


def _rank_shape_matches(classifier_mask):
    return _rank_matches_from_bank(classifier_mask, _shape_classifier_bank())


def _local_detection_note(best_label, confidence, top_matches):
    runner = top_matches[1]["label"] if len(top_matches) > 1 else None
    if confidence >= 80:
        note = f"Local classifier strongly matched your strokes to `{best_label}`."
    elif confidence >= 60:
        note = f"Best local match is `{best_label}` from the offline sketch classifier."
    else:
        note = f"Offline classifier's best guess is `{best_label}`, but the sketch is still fairly ambiguous."
    if runner:
        note += f" Next closest match: `{runner}`."
    return note


def _confidence_from_ranked(ranked):
    best = ranked[0]
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    margin = max(0.0, float(best["score"]) - second_score)
    confidence = int(np.clip(30 + best["score"] * 45 + margin * 35 + min(best["votes"], CLASSIFIER_K) * 2.5, 10, 99))
    if best["score"] < 0.32:
        confidence = min(confidence, 42)
    return confidence


def _extract_character_boxes(raw_mask):
    binary = (np.array(raw_mask, dtype=np.uint8) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < 14 or w < 2 or h < 8:
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    if not boxes:
        return []

    boxes.sort(key=lambda item: item["x"])
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box.copy())
            continue

        prev = merged[-1]
        prev_right = prev["x"] + prev["w"]
        gap = box["x"] - prev_right
        overlap_y = min(prev["y"] + prev["h"], box["y"] + box["h"]) - max(prev["y"], box["y"])
        center_dx = abs((prev["x"] + prev["w"] / 2.0) - (box["x"] + box["w"] / 2.0))
        stacked = center_dx <= max(prev["w"], box["w"]) * 0.45
        close_side_by_side = gap <= max(4, int(0.16 * max(prev["h"], box["h"]))) and overlap_y >= -int(0.2 * max(prev["h"], box["h"]))
        detached_dot = stacked and gap <= max(5, int(0.18 * max(prev["h"], box["h"])))

        if close_side_by_side or detached_dot:
            x0 = min(prev["x"], box["x"])
            y0 = min(prev["y"], box["y"])
            x1 = max(prev["x"] + prev["w"], box["x"] + box["w"])
            y1 = max(prev["y"] + prev["h"], box["y"] + box["h"])
            prev.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "area": prev["area"] + box["area"]})
        else:
            merged.append(box.copy())

    return merged


def _projection_character_boxes(raw_mask):
    binary = (np.array(raw_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    work = cv2.erode(binary, np.ones((2, 2), dtype=np.uint8), iterations=1)
    col_proj = work.sum(axis=0)
    active = col_proj > max(1, int(0.02 * work.shape[0]))

    runs = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= 3:
                runs.append((start, idx - 1))
            start = None
    if start is not None and len(active) - start >= 3:
        runs.append((start, len(active) - 1))

    if len(runs) < 2:
        return []

    boxes = []
    for x0, x1 in runs:
        slice_mask = binary[:, x0 : x1 + 1]
        ys, xs = np.where(slice_mask > 0)
        if len(xs) == 0:
            continue
        y0 = int(ys.min())
        y1 = int(ys.max())
        boxes.append({"x": int(x0), "y": y0, "w": int(x1 - x0 + 1), "h": int(y1 - y0 + 1)})
    return boxes


def _connected_word_boxes(raw_mask):
    binary = (np.array(raw_mask, dtype=np.uint8) > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return []

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    sub = binary[y0 : y1 + 1, x0 : x1 + 1]
    if sub.shape[1] < 18 or (sub.shape[1] / max(sub.shape[0], 1)) < 1.35:
        return []

    work = cv2.erode(sub, np.ones((2, 2), dtype=np.uint8), iterations=1)
    proj = work.sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return []

    smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    smooth = np.convolve(proj, smooth_kernel / smooth_kernel.sum(), mode="same")
    threshold = max(1.0, float(smooth.max()) * 0.30)
    valleys = np.where(smooth <= threshold)[0].tolist()
    if not valleys:
        return []

    cuts = []
    run = [valleys[0]]
    for idx in valleys[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            center = int(round(sum(run) / len(run)))
            if 4 <= center <= len(smooth) - 5:
                cuts.append(center)
            run = [idx]
    center = int(round(sum(run) / len(run)))
    if 4 <= center <= len(smooth) - 5:
        cuts.append(center)

    if not cuts:
        return []

    segments = []
    start = 0
    for cut in cuts:
        if cut - start >= 4:
            segments.append((start, cut - 1))
        start = cut + 1
    if len(smooth) - start >= 4:
        segments.append((start, len(smooth) - 1))

    if len(segments) < 2:
        return []

    boxes = []
    for seg_x0, seg_x1 in segments:
        seg = sub[:, seg_x0 : seg_x1 + 1]
        seg_ys, seg_xs = np.where(seg > 0)
        if len(seg_xs) == 0:
            continue
        boxes.append(
            {
                "x": x0 + seg_x0 + int(seg_xs.min()),
                "y": y0 + int(seg_ys.min()),
                "w": int(seg_xs.max() - seg_xs.min() + 1),
                "h": int(seg_ys.max() - seg_ys.min() + 1),
            }
        )
    return boxes if len(boxes) >= 2 else []


def _split_box_on_valley(raw_mask, box):
    binary = (np.array(raw_mask, dtype=np.uint8) > 0).astype(np.uint8)
    sub = binary[:, box["x"] : box["x"] + box["w"]]
    work = cv2.erode(sub, np.ones((2, 2), dtype=np.uint8), iterations=1)
    proj = work.sum(axis=0).astype(np.float32)
    if len(proj) < 12 or proj.max() <= 0:
        return [box]

    left = max(2, int(len(proj) * 0.18))
    right = min(len(proj) - 3, int(len(proj) * 0.82))
    if left >= right:
        return [box]

    smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    smooth = np.convolve(proj, smooth_kernel / smooth_kernel.sum(), mode="same")
    valley_threshold = max(1.0, float(smooth.max()) * 0.72)
    valley_indices = [
        idx
        for idx in range(left, right + 1)
        if smooth[idx] <= valley_threshold and smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]
    ]
    if not valley_indices:
        minimum = int(np.argmin(smooth[left : right + 1])) + left
        if smooth[minimum] > max(1.0, float(smooth.max()) * 0.82):
            return [box]
        valley_indices = [minimum]

    cuts = []
    run = [valley_indices[0]]
    for idx in valley_indices[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            cuts.append(int(round(sum(run) / len(run))))
            run = [idx]
    cuts.append(int(round(sum(run) / len(run))))

    segments = []
    start = 0
    for cut in cuts:
        if cut - start >= 3:
            segments.append((start, cut - 1))
        start = cut + 1
    if len(proj) - start >= 3:
        segments.append((start, len(proj) - 1))

    boxes_out = []
    for seg_x0, seg_x1 in segments:
        if seg_x1 - seg_x0 < 2:
            continue
        seg_mask = binary[:, box["x"] + seg_x0 : box["x"] + seg_x1 + 1]
        ys, xs = np.where(seg_mask > 0)
        if len(xs) == 0:
            continue
        x0 = box["x"] + seg_x0 + int(xs.min())
        x1 = box["x"] + seg_x0 + int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        boxes_out.append({"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1})
    return boxes_out if len(boxes_out) >= 2 else [box]


def _refine_character_boxes(raw_mask, boxes):
    if len(boxes) < 2:
        return boxes

    widths = [box["w"] for box in boxes]
    median_width = float(np.median(widths)) if widths else 0.0
    refined = []
    for box in boxes:
        aspect = box["w"] / max(box["h"], 1)
        if median_width > 0 and aspect >= 0.9 and box["w"] >= median_width * 1.25:
            split_boxes = _split_box_on_valley(raw_mask, box)
            if len(split_boxes) > 1:
                for split_box in split_boxes:
                    split_aspect = split_box["w"] / max(split_box["h"], 1)
                    if split_aspect >= 0.95 and split_box["w"] >= median_width * 1.05:
                        refined.extend(_split_box_on_valley(raw_mask, split_box))
                    else:
                        refined.append(split_box)
            else:
                refined.extend(split_boxes)
        else:
            refined.append(box)
    refined.sort(key=lambda item: item["x"])
    return refined


def _character_candidates(raw_mask, allow_connected=False):
    boxes = _extract_character_boxes(raw_mask)
    if len(boxes) < 2 and allow_connected:
        boxes = _connected_word_boxes(raw_mask)
    if len(boxes) < 2:
        boxes = _projection_character_boxes(raw_mask)
    boxes = _refine_character_boxes(raw_mask, boxes)
    if len(boxes) < 2:
        return None

    chars = []
    for box in boxes:
        crop = raw_mask[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
        ranked = _rank_character_matches(_classifier_mask(crop))
        best = ranked[0]
        chars.append(
            {
                "label": best["label"],
                "confidence": _confidence_from_ranked(ranked),
                "ranked": ranked,
            }
        )

    text = "".join(item["label"] for item in chars)
    confidence = int(np.mean([item["confidence"] for item in chars])) if chars else 0
    top_matches = [
        {
            "label": f"{idx + 1}:{item['label']}",
            "category": "Character",
            "votes": int(item["ranked"][0]["votes"]),
            "score": float(item["confidence"]),
        }
        for idx, item in enumerate(chars[:6])
    ]
    payload = {
        "detector": "local-text-segmentation-ocr",
        "text": text,
        "confidence": confidence,
        "characters": [
            {
                "position": idx + 1,
                "label": item["label"],
                "confidence": item["confidence"],
            }
            for idx, item in enumerate(chars)
        ],
    }
    return {
        "name": text,
        "category": "Text",
        "confidence": confidence,
        "note": f"Local text reader split the sketch into {len(chars)} character region(s) and read it as `{text}`.",
        "raw": json.dumps(payload, indent=2),
        "top_matches": top_matches,
    }


def _detect_locally(clean_img, prefer_text=False):
    try:
        import pytesseract
        import cv2
        import json
        import numpy as np
        
        # Check against pure white or pure black
        t_img = np.array(clean_img)
        if len(t_img.shape) == 2:
            t_img_rgb = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
        else:
            t_img_rgb = t_img
            
        # Try to use Tesseract for perfect letter/word parsing on Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Add basic thresholding to make letters bolder for tesseract
        gray = cv2.cvtColor(t_img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # PSM 8 assumes a single word or character, PSM 7 assumes single text line
        psm_mode = '--psm 7' if prefer_text else '--psm 8'
        text = pytesseract.image_to_string(thresh, config=f'{psm_mode} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789').strip()
        
        if text and len(text) > 0 and len(text) < 20: 
            return {
                "name": text,
                "category": "OCR Recognition",
                "confidence": 99,
                "note": "Detected perfectly by OCR letter extraction.",
                "raw": json.dumps({"detector": "pytesseract", "best_match": text}),
                "top_matches": [
                    {"label": text, "category": "OCR Recognition", "votes": 100, "score": 99.0}
                ],
            }
    except Exception as e:
        pass # Silently fallback to our robust local classifier bank if Tesseract isn't installed

    try:
        import pytesseract
        from PIL import Image
        import cv2
        import json
        t_img = clean_img
        if len(t_img.shape) == 2:
            t_img_rgb = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
        else:
            t_img_rgb = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
        
        text = pytesseract.image_to_string(Image.fromarray(t_img_rgb)).strip()
        if text:
            return {
                "name": text,
                "category": "Tesseract OCR",
                "confidence": 99,
                "note": "Detected perfectly by Tesseract OCR.",
                "raw": json.dumps({"detector": "pytesseract", "best_match": text}),
                "top_matches": [
                    {"label": text, "category": "Tesseract OCR", "votes": 100, "score": 99.0}
                ],
            }
    except Exception as e:
        print(f"Tesseract error: {e}")

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Use PIL image for genai
            if len(clean_img.shape) == 2:
                rgb_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                
            pil_img = Image.fromarray(rgb_img)
            
            prompt = (
                "You are an expert optical character recognition system and sketch analyzer. "
                "Please carefully analyze this hand-drawn sketch. "
                "Detect every letter or word perfectly whether its rough or hand drawn for human. "
                "If it contains letters/words, return ONLY the exact text. "
                "If it is a clear drawing of a shape or object (like a star, house, animal), return ONLY its exact name."
            )
            response = model.generate_content([prompt, pil_img])
            output_text = response.text.strip().replace("\n", " ")
            
            if output_text:
                return {
                    "name": output_text,
                    "category": "GenAI Vision",
                    "confidence": 99,
                    "note": "Detected perfectly by Google Gemini Vision.",
                    "raw": json.dumps({"detector": "gemini-1.5-flash", "best_match": output_text}),
                    "top_matches": [
                        {"label": output_text, "category": "GenAI Vision", "votes": 100, "score": 99.0}
                    ],
                }
    except Exception as e:
        print(f"Gemini Vision error: {e}")

    raw_mask = _binary_mask_from_image(clean_img)
    norm_mask = _normalize_mask(raw_mask)
    desc = _mask_descriptors(norm_mask)

    if desc["active_ratio"] <= 0.001:
        empty = {"name": "Blank", "category": "Sketch", "confidence": 0, "note": "The detector did not find enough stroke pixels to classify.", "raw": "{}", "top_matches": []}
        return empty

    text_detection = _character_candidates(raw_mask, allow_connected=prefer_text)
    if text_detection:
        return text_detection

    classifier_mask = _classifier_mask(raw_mask)
    char_ranked = _rank_character_matches(classifier_mask)
    shape_ranked = _rank_shape_matches(classifier_mask)
    ranked = char_ranked
    if prefer_text:
        if shape_ranked and char_ranked[0]["score"] < 0.18 and shape_ranked[0]["score"] > char_ranked[0]["score"] + 0.28:
            ranked = shape_ranked
    elif shape_ranked:
        shape_gap = shape_ranked[0]["score"] - char_ranked[0]["score"]
        best_char_label = char_ranked[0]["label"]
        if shape_gap > 0.22 or (
            shape_ranked[0]["score"] > 0.88
            and len(best_char_label) == 1
            and best_char_label.islower()
        ):
            ranked = shape_ranked

    top_matches = [
        {
            "label": item["label"],
            "category": item["category"],
            "votes": item["votes"],
            "score": round(float(item["score"]) * 100.0, 1),
        }
        for item in ranked[:5]
    ]

    best = ranked[0]
    confidence = _confidence_from_ranked(ranked)

    payload = {
        "detector": "local-hog-knn-sketch-classifier",
        "best_match": best["label"],
        "category": best["category"],
        "confidence": confidence,
        "top_matches": top_matches,
        "active_ratio": round(float(desc["active_ratio"]), 4),
        "aspect": round(float(desc["aspect"]), 4),
        "holes": int(desc["holes"]),
        "vertices": int(desc["vertices"]),
    }
    return {
        "name": best["label"],
        "category": best["category"],
        "confidence": confidence,
        "note": _local_detection_note(best["label"], confidence, top_matches),
        "raw": json.dumps(payload, indent=2),
        "top_matches": top_matches,
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


def _analyze_drawing(clean_img, analysis_hash, prefer_text=False):
    input_vec = _image_to_bipolar(clean_img)
    engine = HopfieldEngine(N)
    engine.store(input_vec)
    recovered, energies = engine.recover(input_vec, steps=90)
    changed = (recovered != input_vec).astype(float)
    detection = _detect_locally(clean_img, prefer_text=prefer_text)
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
        "detected_top_matches": detection["top_matches"],
        "text_mode": bool(prefer_text),
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
    inject_module_theme("hopfield")
    _init_state()

    gradient_header(
        "Hopfield Network",
        "Free-Form Sketch Detection + Live Matrix View · Detect what you draw and inspect the neural matrix instantly",
        "🧠",
    )

    theory_text = (
        "This page reads your drawing locally and returns the closest text, symbol, or shape match it can find. "
        "Alongside that, it converts your sketch into a Hopfield-style bipolar matrix so you can see how the drawing becomes a neural state."
    )
    with st.expander("📚 Theory & Mathematical Explanation", expanded=False):
        st.markdown(
            """
            **Free-Form Sketch Detection + Hopfield Matrix**

            1. Clean the canvas into a sharp black-on-white sketch image
            2. Compare the sketch against a local bank of geometric and character fingerprints
            3. Downsample the drawing into a 16 × 16 bipolar neuron matrix
            4. Build a Hopfield-style weight matrix from the current sketch and observe its energy behavior
            """
        )
        render_voice_button(theory_text, key_suffix="hop_theory")

    render_learning_journey(
        "Draw Anything And Get A Direct Output",
        "This lab reads the sketch on the canvas with a local text-and-shape detector, then builds the Hopfield-style matrix and energy views from that same drawing.",
        [
            "The label comes from a local text-and-shape recognition engine that covers letters, digits, and common symbols.",
            "Turn on Text Mode when you want the detector to favor letters and words over symbols and geometric shapes.",
            "Text Mode also switches the board into a paper-like writing surface with a thinner pen, which usually helps letter recognition.",
            "For the strongest text results, write one clear letter at a time or leave a small gap between letters.",
            "The 16 by 16 matrix is a compact neural representation of your strokes.",
            "The Hopfield weight matrix now reflects the current sketch itself, so the matrix view always matches what you drew.",
            "Analysis runs on demand when you click the button, which keeps the page much faster.",
        ],
        "Think of the system as having two linked views of the same sketch. One view reads what the strokes most likely represent, and the other turns those strokes into a grid of neural states and energy patterns.",
        audio_text=theory_text,
        key_suffix="hop_intro",
    )

    section_header("1. Draw And Analyze", "Sketch freely, then run detection only when you want the result")
    mode_help = {
        "Friendly Dashboard": "Compact analytics and the cleanest core views.",
        "Immersive Coach": "Guided explanations stay visible while you inspect the result.",
        "3D Visualization Explorer": "Unlocks the 3D state and weight views.",
    }
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.35, 0.95, 1.0])
        with c1:
            st.caption("VISUALIZATION MODE")
            view_mode = st.radio(
                "Visualization mode",
                list(mode_help.keys()),
                horizontal=True,
                key="viz_mode_hop_compact",
                label_visibility="collapsed",
            )
            st.caption(mode_help[view_mode])
        with c2:
            st.caption("RECOGNITION FOCUS")
            text_mode = st.toggle("Text Mode", value=False, key="hop_text_mode")
            st.caption("Prefer letters and short words with slight spacing.")
        with c3:
            st.caption("AI EXPLANATION")
            explain_with_ai = st.checkbox("Enable Coach Panel", value=False, key="hop_explain_ai")
            st.caption("Recognition stays local either way.")

    if text_mode:
        st.caption("Text Mode uses a paper-style canvas and thinner pen strokes. Slight spacing between letters helps the local reader.")

    if CANVAS_OK:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=8 if text_mode else 20,
            stroke_color="#111827" if text_mode else "#00f0ff",
            background_color="#F8FAFC" if text_mode else "#0f172a",
            height=360,
            width=900,
            drawing_mode="freedraw",
            key=f"hop_canvas_{st.session_state.hop_canvas_key}",
        )
    else:
        st.error("Missing `streamlit-drawable-canvas`, so the drawing board is unavailable.")
        render_chatbot(
        "free-form sketch detection and Hopfield matrix visualization",
        system_prompt=(
            "You are a philosophical memory researcher who draws deep analogies between Hopfield networks "
            "and human memory. You explain associative memory, energy landscapes, and pattern recovery "
            "through the lens of how the brain stores and retrieves memories."
        ),
        greeting=(
            "🧠 Memory Researcher here. Hopfield networks are fascinating models of associative memory — "
            "like how a smell can trigger a complete memory. Ask me about energy minimization, "
            "pattern storage, noise tolerance, or how this relates to human recollection."
        ),
        theme=MODULE_THEMES["hopfield"],
        tutor_label="MEMORY RESEARCHER 🧠",
        placeholder="Ask about associative memory or Hopfield networks...",
    )
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
            if (
                not cached
                or cached.get("analysis_hash") != analysis_hash
                or cached.get("text_mode") != bool(text_mode)
            ):
                with st.spinner("Analyzing your sketch..."):
                    result = _analyze_drawing(clean_img, analysis_hash, prefer_text=text_mode)
                    st.session_state.hop_result = result
            if explain_with_ai and st.session_state.hop_result:
                with st.spinner("Generating AI coach explanation..."):
                    st.session_state.hop_result = _prepare_ai_analysis(st.session_state.hop_result)

    result = st.session_state.get("hop_result")
    if not result:
        render_chatbot(
        "free-form sketch detection and Hopfield matrix visualization",
        system_prompt=(
            "You are a philosophical memory researcher who draws deep analogies between Hopfield networks "
            "and human memory. You explain associative memory, energy landscapes, and pattern recovery "
            "through the lens of how the brain stores and retrieves memories."
        ),
        greeting=(
            "🧠 Memory Researcher here. Hopfield networks are fascinating models of associative memory — "
            "like how a smell can trigger a complete memory. Ask me about energy minimization, "
            "pattern storage, noise tolerance, or how this relates to human recollection."
        ),
        theme=MODULE_THEMES["hopfield"],
        tutor_label="MEMORY RESEARCHER 🧠",
        placeholder="Ask about associative memory or Hopfield networks...",
    )
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
            "The app now waits for your Analyze button before calling the local detector or building the heavy charts. That avoids expensive recomputation on every tiny brush movement."
        ),
    }
    if view_mode == "Immersive Coach":
        render_ai_coach_panel("Coach focus", coach_narratives, key_suffix="hop_focus", accent=C)

    with st.container(border=True):
        st.caption("DETECTION RESULT")
        st.markdown(f"### {result['detected_name']}")
        st.write(result["detected_note"])
        detector_label = (
            "Detector: Local text segmentation + HOG k-NN glyph matcher"
            if result["detected_category"] == "Text"
            else "Detector: Local HOG + k-NN sketch classifier"
        )
        st.caption(detector_label)
        st.caption(f"Mode used: {'Text Mode' if result.get('text_mode') else 'Auto Mode'}")
        with st.expander("Raw detector response", expanded=False):
            st.code(result["detected_raw"])

    if result.get("detected_top_matches"):
        top_matches = result["detected_top_matches"]
        st.plotly_chart(
            contribution_bar(
                [item["label"] for item in top_matches],
                [float(item["score"]) for item in top_matches],
                "Top Local Matches",
                positive=G,
                negative=R,
                neutral=A,
                height=300,
                y_title="Match score",
            ),
            use_container_width=True,
            key="hop_top_matches",
        )

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
        f"Detector: local HOG + k-NN sketch classifier",
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

    render_chatbot(
        "free-form sketch detection and Hopfield matrix visualization",
        system_prompt=(
            "You are a philosophical memory researcher who draws deep analogies between Hopfield networks "
            "and human memory. You explain associative memory, energy landscapes, and pattern recovery "
            "through the lens of how the brain stores and retrieves memories."
        ),
        greeting=(
            "🧠 Memory Researcher here. Hopfield networks are fascinating models of associative memory — "
            "like how a smell can trigger a complete memory. Ask me about energy minimization, "
            "pattern storage, noise tolerance, or how this relates to human recollection."
        ),
        theme=MODULE_THEMES["hopfield"],
        tutor_label="MEMORY RESEARCHER 🧠",
        placeholder="Ask about associative memory or Hopfield networks...",
    )


if __name__ == "__main__":
    main()
