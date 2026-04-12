from __future__ import annotations

from typing import Dict


HAND_SHAPE_MEANINGS = {
    "Auto / unsure": "No hand-shape override was supplied, so the reading leans on the detected line pattern.",
    "Earth": "Earth-hand symbolism leans toward steadiness, practicality, and a preference for concrete results.",
    "Air": "Air-hand symbolism leans toward analysis, curiosity, communication, and quick mental movement.",
    "Fire": "Fire-hand symbolism leans toward initiative, visibility, appetite for action, and a direct style.",
    "Water": "Water-hand symbolism leans toward sensitivity, intuition, imagination, and emotional depth.",
}

DOMINANT_HAND_MEANINGS = {
    "Right": "The right hand is traditionally read as the current path, chosen habits, and outward expression.",
    "Left": "The left hand is traditionally read as inherited tendencies, inner wiring, and baseline temperament.",
    "Both / unsure": "With no clear dominant hand selected, this reading blends inner tendencies with present-day behavior.",
}

LINE_DEPTH_MEANINGS = {
    "Faint": "Fainter lines are usually read as sensitivity, subtle expression, and a style that changes with context.",
    "Medium": "Medium line depth is usually read as balance: steady traits without excessive rigidity.",
    "Deep": "Deeper lines are usually read as intensity, consistency, and traits that show up strongly in day-to-day life.",
}

BREAK_MEANINGS = {
    "None": "Few visible breaks usually point to a steadier rhythm and fewer abrupt resets.",
    "A few": "A few visible breaks often suggest transition points, course corrections, or life chapters that redirect focus.",
    "Many": "Many visible breaks often suggest reinvention, strong adaptability, or a life pattern shaped by change.",
}

FATE_LINE_MEANINGS = {
    "Not visible": "A faint or absent fate line is traditionally read as a self-directed path rather than one fixed by a single script.",
    "Faint": "A faint fate line often points to a career path shaped by flexibility and evolving priorities.",
    "Clear": "A clear fate line often points to a stronger sense of vocation, structure, or long-range direction.",
    "Strong": "A strong fate line often points to pronounced ambition, persistence, and a life path organized around work or mission.",
}

SUN_LINE_MEANINGS = {
    "Not visible": "A faint or absent sun line is traditionally read as motivation coming more from meaning than spotlight.",
    "Faint": "A faint sun line often suggests quiet creativity or recognition that grows gradually.",
    "Clear": "A clear sun line often suggests visible talent, creative confidence, or a stronger desire to be appreciated for quality.",
}


def _safe_ratio(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(value) / float(total)


def _band(value: float, low: float, high: float, labels):
    if value >= high:
        return labels[2]
    if value >= low:
        return labels[1]
    return labels[0]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _prominence_label(ratio: float) -> str:
    return _band(ratio, 0.30, 0.37, ("subtle", "balanced", "dominant"))


def _curvature_label(curvature: float) -> str:
    return _band(curvature, 1.10, 1.24, ("straight", "balanced", "curved"))


def _mental_style(head_curvature: float, head_angle: float) -> str:
    if head_curvature >= 1.22 or abs(head_angle) >= 18:
        return "imaginative and intuitive"
    if head_curvature >= 1.10 or abs(head_angle) >= 10:
        return "flexible and adaptive"
    return "structured and practical"


def _heart_style(heart_curvature: float, heart_ratio: float) -> str:
    if heart_ratio >= 0.37 or heart_curvature >= 1.22:
        return "openly expressive"
    if heart_ratio >= 0.30 or heart_curvature >= 1.10:
        return "warm but measured"
    return "private and steady"


def _life_style(life_curvature: float, life_ratio: float) -> str:
    if life_ratio >= 0.37 or life_curvature >= 1.22:
        return "expansive and experience-seeking"
    if life_ratio >= 0.30 or life_curvature >= 1.10:
        return "balanced and resilient"
    return "steady and energy-conscious"


def _line_reading(line_name: str, ratio: float, curvature: float, angle: float) -> Dict[str, str]:
    prominence = _prominence_label(ratio)
    shape = _curvature_label(curvature)

    if line_name == "Life":
        emphasis = (
            "vitality, pacing, and how the person roots themselves in everyday life"
        )
        detail = (
            f"The life line reads as {prominence} with a {shape} flow. "
            f"Traditional palmistry would read that as a { _life_style(curvature, ratio) } pattern."
        )
    elif line_name == "Head":
        emphasis = "learning style, problem-solving, and decision habits"
        detail = (
            f"The head line reads as {prominence} with a {shape} flow. "
            f"That points toward a mind that feels { _mental_style(curvature, angle) }."
        )
    else:
        emphasis = "emotional expression, bonds, and how affection is shown"
        detail = (
            f"The heart line reads as {prominence} with a {shape} flow. "
            f"That usually reads as an { _heart_style(curvature, ratio) } emotional style."
        )

    return {
        "line": line_name,
        "prominence": prominence,
        "shape": shape,
        "detail": detail,
        "emphasis": emphasis,
    }


def _infer_hand_shape(dominant_line: str, avg_curvature: float) -> str:
    if dominant_line == "Head":
        return "Air" if avg_curvature < 1.16 else "Water"
    if dominant_line == "Heart":
        return "Water" if avg_curvature >= 1.18 else "Fire"
    return "Earth" if avg_curvature < 1.12 else "Fire"


def build_palm_report(features: Dict[str, float], observations: Dict[str, str] | None = None) -> Dict[str, object]:
    observations = observations or {}

    life_length = float(features.get("life_length", 0.0))
    head_length = float(features.get("head_length", 0.0))
    heart_length = float(features.get("heart_length", 0.0))
    total_length = life_length + head_length + heart_length

    line_ratios = {
        "Life": _safe_ratio(life_length, total_length),
        "Head": _safe_ratio(head_length, total_length),
        "Heart": _safe_ratio(heart_length, total_length),
    }
    dominant_line = max(line_ratios, key=line_ratios.get) if total_length > 0 else "Unknown"

    avg_curvature = (
        float(features.get("life_curvature", 0.0))
        + float(features.get("head_curvature", 0.0))
        + float(features.get("heart_curvature", 0.0))
    ) / 3.0

    visible_lines = sum(
        1
        for length in (life_length, head_length, heart_length)
        if length > 20
    )
    detection_quality = min(0.95, _clamp01(0.22 + visible_lines * 0.19 + min(total_length / 900.0, 0.42)))

    hand_shape = observations.get("hand_shape", "Auto / unsure")
    if hand_shape == "Auto / unsure":
        inferred_shape = _infer_hand_shape(dominant_line, avg_curvature)
        hand_shape_label = f"{inferred_shape} (inferred)"
        hand_shape_note = HAND_SHAPE_MEANINGS[inferred_shape]
    else:
        hand_shape_label = hand_shape
        hand_shape_note = HAND_SHAPE_MEANINGS.get(hand_shape, HAND_SHAPE_MEANINGS["Auto / unsure"])

    dominant_hand = observations.get("dominant_hand", "Both / unsure")
    line_depth = observations.get("line_depth", "Medium")
    major_breaks = observations.get("major_breaks", "A few")
    fate_line = observations.get("fate_line", "Faint")
    sun_line = observations.get("sun_line", "Faint")

    life_reading = _line_reading(
        "Life",
        line_ratios["Life"],
        float(features.get("life_curvature", 0.0)),
        float(features.get("life_angle", 0.0)),
    )
    head_reading = _line_reading(
        "Head",
        line_ratios["Head"],
        float(features.get("head_curvature", 0.0)),
        float(features.get("head_angle", 0.0)),
    )
    heart_reading = _line_reading(
        "Heart",
        line_ratios["Heart"],
        float(features.get("heart_curvature", 0.0)),
        float(features.get("heart_angle", 0.0)),
    )

    shared_notes = []
    if int(features.get("life_head_intersection", 0)) > 0:
        shared_notes.append(
            "The life and head lines appear connected early on, which is traditionally read as a careful start and decisions shaped by responsibility."
        )
    if int(features.get("head_heart_intersection", 0)) > 0:
        shared_notes.append(
            "The head and heart lines appear to overlap in places, suggesting that logic and emotion often influence each other instead of staying separate."
        )
    if int(features.get("life_heart_intersection", 0)) > 0:
        shared_notes.append(
            "The life and heart lines show some overlap, which is often read as deep personal investment in family or close bonds."
        )
    if not shared_notes:
        shared_notes.append(
            "The three major lines look fairly distinct, which usually reads as clearer boundaries between energy, thought, and feeling."
        )

    career_shift = "Yes" if abs(float(features.get("head_angle", 0.0))) > 10 and int(features.get("life_head_intersection", 0)) > 0 else "No"

    mindset_theme = (
        f"The head line is {head_reading['prominence']}, so the reading leans { _mental_style(float(features.get('head_curvature', 0.0)), float(features.get('head_angle', 0.0))) }."
    )
    relationship_theme = (
        f"The heart line feels { _heart_style(float(features.get('heart_curvature', 0.0)), line_ratios['Heart']) }, so relationships are read through warmth, pacing, and trust."
    )
    energy_theme = (
        f"The life line suggests a { _life_style(float(features.get('life_curvature', 0.0)), line_ratios['Life']) } rhythm."
    )

    career_theme = (
        f"{FATE_LINE_MEANINGS[fate_line]} "
        f"{'The scan also hints at a meaningful career redirection pattern.' if career_shift == 'Yes' else 'The scan reads as a steadier long-run work rhythm.'}"
    )

    visibility_theme = SUN_LINE_MEANINGS[sun_line]
    stability_theme = BREAK_MEANINGS[major_breaks]
    depth_theme = LINE_DEPTH_MEANINGS[line_depth]
    dominant_hand_theme = DOMINANT_HAND_MEANINGS[dominant_hand]

    summary = (
        f"This reading is led by the {dominant_line.lower()} line, so the strongest theme is "
        f"{head_reading['emphasis'] if dominant_line == 'Head' else heart_reading['emphasis'] if dominant_line == 'Heart' else life_reading['emphasis']}. "
        f"{mindset_theme} {relationship_theme} {energy_theme} "
        f"{hand_shape_note} {depth_theme}"
    )

    practical_focus = [
        career_theme,
        visibility_theme,
        stability_theme,
        dominant_hand_theme,
    ]

    guidance = [
        "Treat the scan as an interpretive reflection, not a fixed prediction.",
        "For a clearer reading, keep the palm flat, centered, and evenly lit so the major lines stay visible.",
        "Use the personalized chat below to ask about love, career, temperament, or decision-making.",
    ]
    if detection_quality < 0.55:
        guidance.insert(
            1,
            "The line extraction is still a little weak. Better contrast and a palm that fills more of the frame will improve the reading.",
        )

    questions = [
        "What does my head line say about how I make decisions?",
        "How does this reading describe my love and relationship style?",
        "What does the scan suggest about career direction or change?",
        "Which part of the reading feels strongest and which looks more subtle?",
    ]

    report = {
        "summary": summary,
        "dominant_line": dominant_line,
        "dominant_strength_pct": round(line_ratios.get(dominant_line, 0.0) * 100, 1) if dominant_line != "Unknown" else 0.0,
        "detection_quality": round(detection_quality, 2),
        "hand_shape_label": hand_shape_label,
        "career_shift_indicator": career_shift,
        "line_readings": [life_reading, head_reading, heart_reading],
        "themes": {
            "mindset": mindset_theme,
            "relationships": relationship_theme,
            "energy": energy_theme,
            "career": career_theme,
            "visibility": visibility_theme,
            "stability": stability_theme,
            "hand_shape": hand_shape_note,
            "dominant_hand": dominant_hand_theme,
            "line_depth": depth_theme,
        },
        "shared_notes": shared_notes,
        "guidance": guidance,
        "questions": questions,
        "line_strengths": line_ratios,
        "observations": {
            "dominant_hand": dominant_hand,
            "hand_shape": hand_shape_label,
            "line_depth": line_depth,
            "major_breaks": major_breaks,
            "fate_line": fate_line,
            "sun_line": sun_line,
        },
    }
    report["chat_context"] = palm_report_to_chat_context(report)
    return report


def palm_report_to_chat_context(report: Dict[str, object]) -> str:
    line_bits = []
    for item in report.get("line_readings", []):
        line_bits.append(
            f"{item['line']} line: {item['detail']} Focus area: {item['emphasis']}."
        )

    theme_map = report.get("themes", {})
    theme_bits = [f"{label.title()}: {text}" for label, text in theme_map.items()]
    shared_notes = " ".join(report.get("shared_notes", []))
    guidance = " ".join(report.get("guidance", []))

    return (
        f"Palm summary: {report.get('summary', '')}\n"
        f"Dominant line: {report.get('dominant_line', 'Unknown')} at {report.get('dominant_strength_pct', 0)}% prominence.\n"
        f"Detection quality: {report.get('detection_quality', 0)}.\n"
        f"Observed profile: {report.get('observations', {})}.\n"
        f"Line notes: {' '.join(line_bits)}\n"
        f"Theme notes: {' '.join(theme_bits)}\n"
        f"Shared pattern notes: {shared_notes}\n"
        f"Guidance notes: {guidance}"
    )


def answer_palm_question(question: str, report: Dict[str, object]) -> str:
    q = question.lower()
    themes = report.get("themes", {})

    if any(word in q for word in ("career", "job", "work", "business", "fate")):
        return (
            f"{themes.get('career', '')} "
            f"{themes.get('stability', '')}"
        ).strip()

    if any(word in q for word in ("love", "heart", "relationship", "marriage", "partner")):
        return (
            f"{themes.get('relationships', '')} "
            f"{report.get('shared_notes', [''])[0]}"
        ).strip()

    if any(word in q for word in ("mind", "study", "decision", "head", "brain", "learn")):
        return themes.get("mindset", report.get("summary", ""))

    if any(word in q for word in ("life", "energy", "health", "vitality")):
        return (
            f"{themes.get('energy', '')} "
            "Palm reading is interpretive tradition, so this should not be used as medical guidance."
        ).strip()

    if any(word in q for word in ("shape", "element", "earth", "air", "fire", "water")):
        return themes.get("hand_shape", report.get("summary", ""))

    if any(word in q for word in ("accurate", "confidence", "clear", "quality")):
        return (
            f"The current scan quality is {report.get('detection_quality', 0):.2f}. "
            "Better lighting, a flatter palm, and a tighter crop usually improve the reading."
        )

    if any(word in q for word in ("dominant", "strongest", "main")):
        return (
            f"The dominant line is {report.get('dominant_line', 'Unknown')} "
            f"at about {report.get('dominant_strength_pct', 0):.1f}% of the detected line pattern."
        )

    return (
        f"{report.get('summary', '')} "
        "Ask about career, love, mindset, or energy for a more focused reading."
    ).strip()
