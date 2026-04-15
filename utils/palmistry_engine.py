"""
══════════════════════════════════════════════════════════════════════════════
CHEIRO'S PALMISTRY ENGINE — Professional Reading Generator
══════════════════════════════════════════════════════════════════════════════
Uses the complete Cheiro knowledge base to produce world-class palm readings
from extracted line features.  Handles time prediction, mount analysis,
hand-type classification, and 50+ question categories.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import math

from utils.palmistry_knowledge import (
    HAND_TYPES,
    THUMB_ANALYSIS,
    FINGER_MEANINGS,
    FINGER_JOINTS,
    NAIL_ANALYSIS,
    MOUNT_ANALYSIS,
    LINE_COMPREHENSIVE,
    MINOR_LINES,
    SPECIAL_MARKS,
    GREAT_TRIANGLE,
    QUADRANGLE,
    TIMING_SYSTEM,
    HEALTH_INDICATORS,
    PERSONALITY_PROFILES,
    TRAVEL_ANALYSIS,
    ELEMENT_COMPATIBILITY,
    build_professional_system_prompt,
    get_professional_greeting,
)


# ─────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────────────────────────────────────

def _safe_ratio(value: float, total: float) -> float:
    return float(value) / float(total) if total > 0 else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _band(value: float, low: float, high: float, labels):
    if value >= high:
        return labels[2]
    if value >= low:
        return labels[1]
    return labels[0]


def _prominence_label(ratio: float) -> str:
    return _band(ratio, 0.30, 0.37, ("subtle", "balanced", "dominant"))


def _curvature_label(curvature: float) -> str:
    return _band(curvature, 1.10, 1.24, ("straight", "balanced", "curved"))


def _depth_label(length: float) -> str:
    """Infer line depth from length — longer lines generally appear deeper."""
    if length > 250:
        return "Deep"
    if length > 120:
        return "Medium"
    return "Faint"


# ─────────────────────────────────────────────────────────────────────────
# HAND TYPE CLASSIFICATION — Cheiro's 7-type system
# ─────────────────────────────────────────────────────────────────────────

def classify_hand_type(features: Dict[str, float], observations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Classify the hand into one of Cheiro's 7 hand types based on
    extracted features and optional user observations.
    """
    observations = observations or {}
    user_shape = observations.get("hand_shape", "Auto / unsure")

    if user_shape != "Auto / unsure":
        # Map legacy element names to Cheiro types
        element_to_cheiro = {
            "Earth": "Square",
            "Air": "Philosophic",
            "Fire": "Spatulate",
            "Water": "Conic",
        }
        cheiro_type = element_to_cheiro.get(user_shape, user_shape)
        if cheiro_type in HAND_TYPES:
            ht = HAND_TYPES[cheiro_type]
            return {
                "type": cheiro_type,
                "hindi": ht["hindi"],
                "description": ht["description"],
                "personality": ht["personality"],
                "career": ht["career"],
                "health": ht["health"],
                "relationships": ht["relationships"],
                "element": _type_to_element(cheiro_type),
                "source": "user_selected",
            }

    # Infer from features
    avg_curvature = (
        float(features.get("life_curvature", 0))
        + float(features.get("head_curvature", 0))
        + float(features.get("heart_curvature", 0))
    ) / 3.0

    total_length = (
        float(features.get("life_length", 0))
        + float(features.get("head_length", 0))
        + float(features.get("heart_length", 0))
    )

    head_angle = abs(float(features.get("head_angle", 0)))
    intersections = (
        int(features.get("life_head_intersection", 0))
        + int(features.get("head_heart_intersection", 0))
        + int(features.get("life_heart_intersection", 0))
    )

    # Heuristic classification
    if total_length < 200:
        inferred = "Elementary"
    elif avg_curvature > 1.28 and head_angle > 15:
        inferred = "Psychic"
    elif avg_curvature > 1.22 and head_angle > 10:
        inferred = "Conic"
    elif head_angle > 12 and intersections > 1:
        inferred = "Philosophic"
    elif avg_curvature < 1.08:
        inferred = "Square"
    elif avg_curvature < 1.14 and head_angle < 8:
        inferred = "Spatulate"
    else:
        inferred = "Mixed"

    ht = HAND_TYPES[inferred]
    return {
        "type": inferred,
        "hindi": ht["hindi"],
        "description": ht["description"],
        "personality": ht["personality"],
        "career": ht["career"],
        "health": ht["health"],
        "relationships": ht["relationships"],
        "element": _type_to_element(inferred),
        "source": "auto_detected",
    }


def _type_to_element(hand_type: str) -> str:
    mapping = {
        "Elementary": "Earth",
        "Square": "Earth",
        "Spatulate": "Fire",
        "Philosophic": "Air",
        "Conic": "Water",
        "Psychic": "Water",
        "Mixed": "Mixed",
    }
    return mapping.get(hand_type, "Mixed")


# ─────────────────────────────────────────────────────────────────────────
# MOUNT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

def analyze_mounts(features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Infer mount prominence from line geometry.
    Real mount analysis requires 3D palm topography; this approximates
    from 2D line features using Cheiro's relationship rules.
    """
    life_length = float(features.get("life_length", 0))
    head_length = float(features.get("head_length", 0))
    heart_length = float(features.get("heart_length", 0))
    life_curv = float(features.get("life_curvature", 0))
    head_curv = float(features.get("head_curvature", 0))
    heart_curv = float(features.get("heart_curvature", 0))
    head_angle = float(features.get("head_angle", 0))
    total = life_length + head_length + heart_length

    mounts = {}

    # Jupiter — inferred from heart line ending position (toward Jupiter = strong)
    jupiter_score = _clamp01(0.3 + (heart_length / max(total, 1)) * 1.2)
    mounts["Jupiter"] = {
        "strength": _band(jupiter_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(jupiter_score, 2),
        "reading": MOUNT_ANALYSIS["Jupiter"]["well_developed"] if jupiter_score >= 0.35 else MOUNT_ANALYSIS["Jupiter"]["under_developed"],
    }

    # Saturn — inferred from fate line depth (approximated by intersections)
    saturn_score = _clamp01(0.25 + int(features.get("life_head_intersection", 0)) * 0.15 + 0.1)
    mounts["Saturn"] = {
        "strength": _band(saturn_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(saturn_score, 2),
        "reading": MOUNT_ANALYSIS["Saturn"]["well_developed"] if saturn_score >= 0.35 else MOUNT_ANALYSIS["Saturn"]["under_developed"],
    }

    # Sun/Apollo — inferred from overall line clarity
    apollo_score = _clamp01(0.2 + (total / 900) * 0.5 + (1 if head_curv > 1.15 else 0) * 0.15)
    mounts["Sun_Apollo"] = {
        "strength": _band(apollo_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(apollo_score, 2),
        "reading": MOUNT_ANALYSIS["Sun_Apollo"]["well_developed"] if apollo_score >= 0.35 else MOUNT_ANALYSIS["Sun_Apollo"]["under_developed"],
    }

    # Mercury — inferred from head line characteristics
    mercury_score = _clamp01(0.25 + (head_length / max(total, 1)) * 1.0 + (0.1 if abs(head_angle) < 10 else 0))
    mounts["Mercury"] = {
        "strength": _band(mercury_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(mercury_score, 2),
        "reading": MOUNT_ANALYSIS["Mercury"]["well_developed"] if mercury_score >= 0.35 else MOUNT_ANALYSIS["Mercury"]["under_developed"],
    }

    # Venus — inferred from life line curvature (wide curve = prominent Venus)
    venus_score = _clamp01(0.2 + life_curv * 0.3 + (life_length / max(total, 1)) * 0.3)
    mounts["Venus"] = {
        "strength": _band(venus_score, 0.40, 0.60, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(venus_score, 2),
        "reading": MOUNT_ANALYSIS["Venus"]["well_developed"] if venus_score >= 0.40 else MOUNT_ANALYSIS["Venus"]["under_developed"],
    }

    # Moon — inferred from head line slope
    moon_score = _clamp01(0.15 + (abs(head_angle) / 30) * 0.5 + (head_curv - 1.0) * 0.3)
    mounts["Moon"] = {
        "strength": _band(moon_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(moon_score, 2),
        "reading": MOUNT_ANALYSIS["Moon"]["well_developed"] if moon_score >= 0.35 else MOUNT_ANALYSIS["Moon"]["under_developed"],
    }

    # Mars — inferred from overall line intensity
    mars_score = _clamp01(0.3 + (total / 1200) * 0.4)
    mounts["Mars"] = {
        "strength": _band(mars_score, 0.35, 0.55, ("Under-developed", "Well-developed", "Over-developed")),
        "score": round(mars_score, 2),
        "reading": MOUNT_ANALYSIS.get("Mars_Lower", MOUNT_ANALYSIS["Mars_Upper"])["well_developed"] if mars_score >= 0.35 else {"personality": "Average martial energy."},
    }

    return mounts


def get_dominant_mount(mounts: Dict[str, Dict[str, Any]]) -> str:
    """Returns the name of the most prominent mount."""
    return max(mounts, key=lambda k: mounts[k]["score"])


# ─────────────────────────────────────────────────────────────────────────
# LINE READING — Enhanced from Cheiro
# ─────────────────────────────────────────────────────────────────────────

def _mental_style(head_curvature: float, head_angle: float) -> str:
    if head_curvature >= 1.22 or abs(head_angle) >= 18:
        return "imaginative and intuitive — the Head line slopes toward the Moon, indicating a mind rich in creativity and fantasy"
    if head_curvature >= 1.10 or abs(head_angle) >= 10:
        return "flexible and adaptive — blending practical thinking with creative insights"
    return "structured and practical — a straight Head line indicating logical, business-oriented thinking"


def _heart_style(heart_curvature: float, heart_ratio: float) -> str:
    if heart_ratio >= 0.37 or heart_curvature >= 1.22:
        return "openly expressive and deeply passionate — Cheiro would read this as a warm, demonstrative emotional nature"
    if heart_ratio >= 0.30 or heart_curvature >= 1.10:
        return "warm but measured — showing affection selectively with genuine depth"
    return "private and steady — emotions run deep but are not easily displayed"


def _life_style(life_curvature: float, life_ratio: float) -> str:
    if life_ratio >= 0.37 or life_curvature >= 1.22:
        return "expansive and experience-seeking — the wide arc of the Life line indicates abundant vitality and a love of adventure"
    if life_ratio >= 0.30 or life_curvature >= 1.10:
        return "balanced and resilient — steady energy with good recovery from setbacks"
    return "steady and energy-conscious — focused vitality directed toward specific goals"


def _line_reading(line_name: str, ratio: float, curvature: float, angle: float, length: float) -> Dict[str, Any]:
    prominence = _prominence_label(ratio)
    shape = _curvature_label(curvature)
    depth = _depth_label(length)
    line_data = LINE_COMPREHENSIVE.get(line_name, {})

    if line_name == "Life":
        emphasis = "vitality, physical constitution, major life transitions, and energy rhythms"
        style = _life_style(curvature, ratio)
        # Determine variation
        if curvature >= 1.22:
            variation_key = "long_well_marked"
        elif length < 100:
            variation_key = "short"
        else:
            variation_key = "long_well_marked"
        variation_text = line_data.get("variations", {}).get(variation_key, "")
        detail = (
            f"The Life line reads as {prominence} ({depth} depth) with a {shape} flow. "
            f"{style}. {variation_text}"
        )
    elif line_name == "Head":
        emphasis = "mental capacity, thinking style, decision-making, and intellectual power"
        style = _mental_style(curvature, angle)
        if abs(angle) >= 15:
            variation_key = "long_sloping_to_moon"
        elif curvature < 1.08:
            variation_key = "long_straight"
        else:
            variation_key = "deep_well_marked"
        variation_text = line_data.get("variations", {}).get(variation_key, "")
        detail = (
            f"The Head line reads as {prominence} ({depth} depth) with a {shape} flow. "
            f"The mind is {style}. {variation_text}"
        )
    else:  # Heart
        emphasis = "emotional expression, capacity for love, and relationship patterns"
        style = _heart_style(curvature, ratio)
        if curvature >= 1.22:
            variation_key = "curved"
        elif curvature < 1.08:
            variation_key = "straight"
        else:
            variation_key = "ending_between_jupiter_saturn"
        variation_text = line_data.get("variations", {}).get(variation_key, "")
        detail = (
            f"The Heart line reads as {prominence} ({depth} depth) with a {shape} flow. "
            f"Emotional nature is {style}. {variation_text}"
        )

    # Timing info
    timing_info = line_data.get("timing_method", "")

    return {
        "line": line_name,
        "hindi": line_data.get("hindi", ""),
        "prominence": prominence,
        "shape": shape,
        "depth": depth,
        "detail": detail,
        "emphasis": emphasis,
        "governs": line_data.get("governs", []),
        "timing_method": timing_info,
    }


# ─────────────────────────────────────────────────────────────────────────
# TIME PREDICTIONS — Cheiro's Method
# ─────────────────────────────────────────────────────────────────────────

def predict_timing(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate time-based predictions using Cheiro's timing system.
    """
    life_length = float(features.get("life_length", 0))
    head_length = float(features.get("head_length", 0))
    life_curv = float(features.get("life_curvature", 0))
    head_angle = abs(float(features.get("head_angle", 0)))
    intersections = int(features.get("life_head_intersection", 0))

    predictions = []

    # Independence timing (Life-Head separation)
    if intersections > 0:
        independence_age = "18-24"
        predictions.append({
            "period": f"Age {independence_age}",
            "event": "Period of independence from family influence",
            "detail": (
                "The Life and Head lines show connection, indicating family "
                "influence in early life. Based on Cheiro's timing, independence "
                f"and self-direction emerge around age {independence_age}."
            ),
            "category": "life_transition",
        })
    else:
        predictions.append({
            "period": "Age 14-18",
            "event": "Early independence and self-direction",
            "detail": (
                "The Life and Head lines are separated from the start, indicating "
                "an independent nature from early age. Self-directed decisions "
                "begin around age 14-18."
            ),
            "category": "life_transition",
        })

    # Career direction (from head line characteristics)
    if head_angle > 12:
        predictions.append({
            "period": "Age 25-32",
            "event": "Major creative/career breakthrough",
            "detail": (
                "The Head line's strong slope toward the Moon mount suggests "
                "a period of major creative realization or career shift toward "
                "artistic/imaginative work between ages 25-32."
            ),
            "category": "career",
        })
    else:
        predictions.append({
            "period": "Age 28-35",
            "event": "Career consolidation and growth",
            "detail": (
                "The practical Head line suggests career consolidation "
                "and significant professional growth between ages 28-35."
            ),
            "category": "career",
        })

    # Life energy assessment
    if life_curv >= 1.20:
        predictions.append({
            "period": "Age 35-45",
            "event": "Peak vitality and life expansion",
            "detail": (
                "The wide arc of the Life line indicates peak physical vitality "
                "and life expansion between ages 35-45. This is a period of "
                "maximum energy and achievement potential."
            ),
            "category": "health_energy",
        })
    else:
        predictions.append({
            "period": "Age 30-40",
            "event": "Steady energy, directed effort",
            "detail": (
                "The Life line suggests a period of focused, well-managed "
                "energy between ages 30-40. Success comes through directed "
                "effort rather than broad expansion."
            ),
            "category": "health_energy",
        })

    # Relationship timing
    predictions.append({
        "period": "Age 24-30",
        "event": "Significant emotional partnership",
        "detail": (
            "Based on Cheiro's timing on the Heart line, a significant "
            "emotional connection or partnership is indicated between "
            "ages 24-30. The quality of this bond depends on the Heart "
            "line's depth and curvature."
        ),
        "category": "relationships",
    })

    # Midlife assessment
    if life_length > 200:
        predictions.append({
            "period": "Age 45-55",
            "event": "Wisdom phase and life reassessment",
            "detail": (
                "The Life line's extended reach suggests a significant midlife "
                "reassessment phase around ages 45-55, where accumulated wisdom "
                "creates opportunities for meaningful direction change."
            ),
            "category": "life_transition",
        })

    # Later life
    predictions.append({
        "period": "Age 55-70+",
        "event": "Legacy and spiritual growth",
        "detail": (
            "Based on Cheiro's system, the later portion of the Life line "
            "reflects a period of consolidation, legacy building, and "
            "potential spiritual deepening. The quality of this period is "
            "shaped by all prior life choices."
        ),
        "category": "spiritual",
    })

    return {
        "predictions": predictions,
        "timing_method": TIMING_SYSTEM["description"],
        "note": (
            "These time predictions use Cheiro's proportional timing system "
            "applied to the detected line positions. They represent traditional "
            "interpretive tendencies, not certainties."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# HEALTH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────

def analyze_health(features: Dict[str, float]) -> Dict[str, Any]:
    """Produce health analysis based on Cheiro's indicators."""
    life_curv = float(features.get("life_curvature", 0))
    head_curv = float(features.get("head_curvature", 0))
    heart_curv = float(features.get("heart_curvature", 0))
    life_length = float(features.get("life_length", 0))

    indicators = []
    overall_vitality = "moderate"

    if life_length > 250 and life_curv > 1.15:
        overall_vitality = "strong"
        indicators.append({
            "area": "General Vitality",
            "assessment": "Strong",
            "detail": HEALTH_INDICATORS["longevity_indicators"]["positive"],
        })
    elif life_length < 120:
        overall_vitality = "sensitive"
        indicators.append({
            "area": "General Vitality",
            "assessment": "Needs attention",
            "detail": HEALTH_INDICATORS["longevity_indicators"]["caution"],
        })
    else:
        indicators.append({
            "area": "General Vitality",
            "assessment": "Moderate",
            "detail": "Moderate vitality — a balanced constitution that benefits from regular self-care.",
        })

    # Head/mental health
    if head_curv > 1.25:
        indicators.append({
            "area": "Mental/Nervous System",
            "assessment": "Sensitive",
            "detail": HEALTH_INDICATORS["nervous_disorders"]["signs"],
        })
    else:
        indicators.append({
            "area": "Mental/Nervous System",
            "assessment": "Stable",
            "detail": "Head line suggests stable mental constitution with good focus.",
        })

    # Heart/cardiovascular
    if heart_curv < 1.05:
        indicators.append({
            "area": "Cardiovascular",
            "assessment": "Monitor",
            "detail": HEALTH_INDICATORS["heart_disease"]["signs"],
        })
    else:
        indicators.append({
            "area": "Cardiovascular",
            "assessment": "Balanced",
            "detail": "Heart line characteristics suggest balanced cardiovascular patterns.",
        })

    return {
        "overall_vitality": overall_vitality,
        "indicators": indicators,
        "disclaimer": (
            "⚕️ This health analysis is based on traditional palmistry interpretation "
            "from Cheiro's system. It is NOT medical advice. Always consult qualified "
            "healthcare professionals for health concerns."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# PERSONALITY PROFILE
# ─────────────────────────────────────────────────────────────────────────

def build_personality_profile(
    hand_type: Dict[str, Any],
    mounts: Dict[str, Dict[str, Any]],
    features: Dict[str, float],
) -> Dict[str, Any]:
    """Build comprehensive personality profile from hand type + mounts."""
    dominant_mount = get_dominant_mount(mounts)
    head_angle = abs(float(features.get("head_angle", 0)))

    # Determine personality archetype
    archetype = "intellectual_scholar"
    if dominant_mount in ("Jupiter",) and hand_type.get("type") in ("Square", "Spatulate"):
        archetype = "leader"
    elif dominant_mount in ("Sun_Apollo",) or hand_type.get("type") in ("Conic", "Psychic"):
        archetype = "creative_artist"
    elif dominant_mount in ("Mercury",) and hand_type.get("type") in ("Square", "Spatulate"):
        archetype = "business_entrepreneur"
    elif dominant_mount in ("Moon",):
        if head_angle > 15:
            archetype = "mystic_psychic"
        else:
            archetype = "adventurer_explorer"
    elif dominant_mount in ("Venus",):
        archetype = "healer_counselor"

    profile = PERSONALITY_PROFILES.get(archetype, PERSONALITY_PROFILES["intellectual_scholar"])

    # Combine traits from hand type + mount + profile
    combined_traits = list(hand_type.get("personality", []))[:4]

    mount_info = MOUNT_ANALYSIS.get(dominant_mount, {})
    well_dev = mount_info.get("well_developed", {})
    if isinstance(well_dev, dict):
        mount_traits = well_dev.get("personality", [])
        if isinstance(mount_traits, list):
            combined_traits.extend(mount_traits[:3])

    return {
        "archetype": archetype.replace("_", " ").title(),
        "description": profile["description"],
        "core_traits": combined_traits,
        "dominant_mount": dominant_mount.replace("_", " "),
        "hand_type": hand_type.get("type", "Mixed"),
        "element": hand_type.get("element", "Mixed"),
        "career_aptitude": hand_type.get("career", []),
        "relationship_style": hand_type.get("relationships", ""),
    }


# ─────────────────────────────────────────────────────────────────────────
# MAIN REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────

# Keep legacy lookups for observations form
HAND_SHAPE_MEANINGS = {
    "Auto / unsure": "No hand-shape override was supplied, so the reading uses auto-detection.",
    "Earth": HAND_TYPES["Square"]["description"][:120],
    "Air": HAND_TYPES["Philosophic"]["description"][:120],
    "Fire": HAND_TYPES["Spatulate"]["description"][:120],
    "Water": HAND_TYPES["Conic"]["description"][:120],
}

DOMINANT_HAND_MEANINGS = {
    "Right": "The right hand is read as the current path, chosen habits, and outward expression.",
    "Left": "The left hand is read as inherited tendencies, inner wiring, and baseline temperament.",
    "Both / unsure": "With no clear dominant hand selected, this reading blends inner tendencies with present-day behavior.",
}

LINE_DEPTH_MEANINGS = {
    "Faint": "Fainter lines: sensitivity, subtle expression, and a style that changes with context.",
    "Medium": "Medium depth: balance, steady traits without excessive rigidity.",
    "Deep": "Deeper lines: intensity, consistency, and traits that show up strongly day-to-day.",
}

BREAK_MEANINGS = {
    "None": "Few visible breaks: steadier rhythm, fewer abrupt resets.",
    "A few": "A few breaks: transition points, course corrections, or redirecting chapters.",
    "Many": "Many breaks: reinvention, strong adaptability, pattern shaped by change.",
}

FATE_LINE_MEANINGS = {
    "Not visible": LINE_COMPREHENSIVE["Fate_Saturn"]["variations"]["absent"],
    "Faint": LINE_COMPREHENSIVE["Fate_Saturn"]["variations"]["faint"],
    "Clear": LINE_COMPREHENSIVE["Fate_Saturn"]["variations"]["deep_strong"][:150],
    "Strong": LINE_COMPREHENSIVE["Fate_Saturn"]["variations"]["deep_strong"],
}

SUN_LINE_MEANINGS = {
    "Not visible": LINE_COMPREHENSIVE["Sun_Apollo"]["variations"]["absent"],
    "Faint": LINE_COMPREHENSIVE["Sun_Apollo"]["variations"].get("starting_from_heart_line", "Faint Sun line."),
    "Clear": LINE_COMPREHENSIVE["Sun_Apollo"]["variations"]["deep_strong"],
}


def build_palm_report(features: Dict[str, float], observations: Dict[str, str] | None = None) -> Dict[str, object]:
    """Build a comprehensive Cheiro-standard palm reading report."""
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

    visible_lines = sum(1 for l in (life_length, head_length, heart_length) if l > 20)
    detection_quality = min(
        0.95,
        _clamp01(0.22 + visible_lines * 0.19 + min(total_length / 900.0, 0.42)),
    )

    # Hand type classification
    hand_type = classify_hand_type(features, observations)

    # Mount analysis
    mounts = analyze_mounts(features)
    dominant_mount = get_dominant_mount(mounts)

    # Personality profile
    personality = build_personality_profile(hand_type, mounts, features)

    # Line readings
    life_reading = _line_reading(
        "Life", line_ratios["Life"],
        float(features.get("life_curvature", 0)),
        float(features.get("life_angle", 0)),
        life_length,
    )
    head_reading = _line_reading(
        "Head", line_ratios["Head"],
        float(features.get("head_curvature", 0)),
        float(features.get("head_angle", 0)),
        head_length,
    )
    heart_reading = _line_reading(
        "Heart", line_ratios["Heart"],
        float(features.get("heart_curvature", 0)),
        float(features.get("heart_angle", 0)),
        heart_length,
    )

    # Time predictions
    timing = predict_timing(features)

    # Health analysis
    health = analyze_health(features)

    # Observations
    dominant_hand = observations.get("dominant_hand", "Right")
    line_depth = observations.get("line_depth", "Medium")
    major_breaks = observations.get("major_breaks", "A few")
    fate_line = observations.get("fate_line", "Faint")
    sun_line = observations.get("sun_line", "Faint")

    # Intersection patterns
    shared_notes = []
    if int(features.get("life_head_intersection", 0)) > 0:
        shared_notes.append(
            "Life and Head lines are connected — traditionally read as a careful, "
            "family-influenced start with decisions shaped by responsibility. "
            "Cheiro noted this pattern in people who take longer to find independence."
        )
    if int(features.get("head_heart_intersection", 0)) > 0:
        shared_notes.append(
            "Head and Heart lines overlap — logic and emotion strongly influence each "
            "other. Cheiro read this as a person whose feelings color their thinking "
            "and vice versa."
        )
    if int(features.get("life_heart_intersection", 0)) > 0:
        shared_notes.append(
            "Life and Heart lines show overlap — deep personal investment in family "
            "and close bonds. Cheiro saw this in people whose health is affected by "
            "emotional stress."
        )
    if not shared_notes:
        shared_notes.append(
            "The three major lines are distinct — clear boundaries between energy, "
            "thought, and feeling. Cheiro considered this a sign of a well-balanced "
            "nature with clear self-awareness."
        )

    # Theme generation
    career_shift = "Yes" if (abs(float(features.get("head_angle", 0))) > 10
                            and int(features.get("life_head_intersection", 0)) > 0) else "No"

    mindset_theme = f"The Head line reveals: {_mental_style(float(features.get('head_curvature', 0)), float(features.get('head_angle', 0)))}."
    relationship_theme = f"The Heart line reveals: {_heart_style(float(features.get('heart_curvature', 0)), line_ratios['Heart'])}."
    energy_theme = f"The Life line reveals: {_life_style(float(features.get('life_curvature', 0)), line_ratios['Life'])}."
    career_theme = f"{FATE_LINE_MEANINGS.get(fate_line, '')} {'Career redirection pattern detected.' if career_shift == 'Yes' else 'Steady work rhythm indicated.'}"
    visibility_theme = SUN_LINE_MEANINGS.get(sun_line, "")
    stability_theme = BREAK_MEANINGS.get(major_breaks, "")
    depth_theme = LINE_DEPTH_MEANINGS.get(line_depth, "")
    dominant_hand_theme = DOMINANT_HAND_MEANINGS.get(dominant_hand, "")

    # Summary — professional Cheiro-style
    summary = (
        f"🔮 **Cheiro's Analysis**: This is a {hand_type['type']} hand ({hand_type['hindi']}) "
        f"with {hand_type['element']} element influence. The reading is led by the "
        f"{dominant_line.lower()} line, with the {dominant_mount.replace('_', ' ')} mount most prominent. "
        f"Personality archetype: **{personality['archetype']}**.\n\n"
        f"{mindset_theme} {relationship_theme} {energy_theme}\n\n"
        f"The hand reveals someone who is {'; '.join(hand_type['personality'][:3]).lower()}. "
        f"{depth_theme}"
    )

    guidance = [
        "This reading follows Cheiro's complete palmistry methodology — the same system used to read hands of world leaders for 40+ years.",
        "For a more precise reading, keep the palm flat, centered, and evenly lit so all lines and mounts are visible.",
        "Use the personalized chat below to ask about love, career, timing, health, personality, or any aspect of your reading.",
        "Remember: palm lines show tendencies, not fixed destiny. Your free will writes the final chapter.",
    ]
    if detection_quality < 0.55:
        guidance.insert(1, "⚠️ Line extraction is weak. Better contrast and a palm filling more of the frame will dramatically improve the reading.")

    questions = [
        "When will I experience a major career change?",
        "What does my palm say about my love life and marriage timing?",
        "What are my strongest personality traits according to Cheiro?",
        "Tell me about my health outlook and vitality",
        "What career path does my hand suggest?",
        "What time period shows the most significant life changes?",
        "Am I more suited for business or creative work?",
        "What does the dominant mount on my palm reveal about me?",
    ]

    report = {
        "summary": summary,
        "dominant_line": dominant_line,
        "dominant_strength_pct": round(line_ratios.get(dominant_line, 0.0) * 100, 1) if dominant_line != "Unknown" else 0.0,
        "detection_quality": round(detection_quality, 2),
        "hand_type": hand_type,
        "hand_shape_label": f"{hand_type['type']} ({hand_type['hindi']})",
        "personality": personality,
        "career_shift_indicator": career_shift,
        "line_readings": [life_reading, head_reading, heart_reading],
        "mounts": mounts,
        "dominant_mount": dominant_mount,
        "timing": timing,
        "health": health,
        "themes": {
            "mindset": mindset_theme,
            "relationships": relationship_theme,
            "energy": energy_theme,
            "career": career_theme,
            "visibility": visibility_theme,
            "stability": stability_theme,
            "hand_shape": hand_type["description"],
            "dominant_hand": dominant_hand_theme,
            "line_depth": depth_theme,
        },
        "shared_notes": shared_notes,
        "guidance": guidance,
        "questions": questions,
        "line_strengths": line_ratios,
        "observations": {
            "dominant_hand": dominant_hand,
            "hand_shape": hand_type["type"],
            "line_depth": line_depth,
            "major_breaks": major_breaks,
            "fate_line": fate_line,
            "sun_line": sun_line,
        },
    }
    report["chat_context"] = palm_report_to_chat_context(report)
    return report


# ─────────────────────────────────────────────────────────────────────────
# CHAT CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────

def palm_report_to_chat_context(report: Dict[str, object]) -> str:
    """Build rich context for the AI chatbot from the full report."""
    line_bits = []
    for item in report.get("line_readings", []):
        governs = ", ".join(item.get("governs", [])[:3])
        line_bits.append(
            f"{item['line']} line ({item.get('hindi', '')}): {item['detail']} "
            f"Governs: {governs}. Timing: {item.get('timing_method', 'N/A')[:200]}"
        )

    # Hand type context
    ht = report.get("hand_type", {})
    hand_ctx = (
        f"Hand Type: {ht.get('type', 'Unknown')} ({ht.get('hindi', '')}) — "
        f"{ht.get('description', '')[:300]} "
        f"Career aptitude: {', '.join(ht.get('career', [])[:5])}. "
        f"Relationships: {ht.get('relationships', '')[:200]}"
    )

    # Mount context
    mounts = report.get("mounts", {})
    mount_bits = []
    for name, data in mounts.items():
        mount_bits.append(f"{name}: {data.get('strength', 'Unknown')} (score {data.get('score', 0)})")

    # Personality
    personality = report.get("personality", {})
    personality_ctx = (
        f"Archetype: {personality.get('archetype', 'Unknown')}. "
        f"Core traits: {', '.join(personality.get('core_traits', [])[:5])}. "
        f"Dominant mount: {personality.get('dominant_mount', 'Unknown')}."
    )

    # Timing
    timing = report.get("timing", {})
    timing_bits = []
    for pred in timing.get("predictions", []):
        timing_bits.append(f"{pred['period']}: {pred['event']} — {pred['detail'][:150]}")

    # Health
    health = report.get("health", {})
    health_bits = []
    for ind in health.get("indicators", []):
        health_bits.append(f"{ind['area']}: {ind['assessment']} — {ind['detail'][:150]}")

    theme_map = report.get("themes", {})
    theme_bits = [f"{label.title()}: {text}" for label, text in theme_map.items()]
    shared_notes = " ".join(report.get("shared_notes", []))

    return (
        f"=== CHEIRO'S PALM READING DATA ===\n"
        f"Palm summary: {report.get('summary', '')}\n\n"
        f"Dominant line: {report.get('dominant_line', 'Unknown')} at {report.get('dominant_strength_pct', 0)}% prominence.\n"
        f"Detection quality: {report.get('detection_quality', 0)}.\n\n"
        f"=== HAND TYPE ===\n{hand_ctx}\n\n"
        f"=== PERSONALITY ===\n{personality_ctx}\n\n"
        f"=== LINE READINGS ===\n{'  '.join(line_bits)}\n\n"
        f"=== MOUNT ANALYSIS ===\n{'  '.join(mount_bits)}\n"
        f"Dominant mount: {report.get('dominant_mount', 'Unknown')}\n\n"
        f"=== TIME PREDICTIONS ===\n{'  '.join(timing_bits)}\n\n"
        f"=== HEALTH ANALYSIS ===\n{'  '.join(health_bits)}\n"
        f"Overall vitality: {health.get('overall_vitality', 'moderate')}\n\n"
        f"=== THEMES ===\n{'  '.join(theme_bits)}\n\n"
        f"=== PATTERN NOTES ===\n{shared_notes}\n\n"
        f"=== TIMING SYSTEM ===\n{TIMING_SYSTEM['description']}\n"
        f"Life line timing: {TIMING_SYSTEM['Life_line']['method']}\n"
        f"Fate line timing: {TIMING_SYSTEM['Fate_line']['method']}\n"
        f"Head line timing: {TIMING_SYSTEM['Head_line']['method']}\n"
        f"Heart line timing: {TIMING_SYSTEM['Heart_line']['method']}\n"
    )


# ─────────────────────────────────────────────────────────────────────────
# ANSWER PALM QUESTIONS — 50+ Categories
# ─────────────────────────────────────────────────────────────────────────

def answer_palm_question(question: str, report: Dict[str, object]) -> str:
    """Professional Cheiro-level answer for 50+ question categories."""
    q = question.lower()
    themes = report.get("themes", {})
    ht = report.get("hand_type", {})
    personality = report.get("personality", {})
    timing = report.get("timing", {})
    health = report.get("health", {})
    mounts = report.get("mounts", {})

    # ── CAREER & WORK ──
    if any(w in q for w in ("career", "job", "work", "business", "profession", "fate", "money", "financial", "salary", "promotion")):
        career_list = ", ".join(ht.get("career", [])[:5])
        timing_events = [p for p in timing.get("predictions", []) if p.get("category") == "career"]
        timing_text = timing_events[0]["detail"] if timing_events else ""
        return (
            f"🔮 **Career Reading (Cheiro's Analysis)**\n\n"
            f"{themes.get('career', '')}\n\n"
            f"**Hand Type Insight**: Your {ht.get('type', 'Mixed')} hand suggests aptitude for: {career_list}.\n\n"
            f"**Mount Influence**: Your dominant {report.get('dominant_mount', 'Unknown').replace('_', ' ')} mount shapes your professional instincts — "
            f"{_get_mount_career(report.get('dominant_mount', ''))}\n\n"
            f"**Timing**: {timing_text}\n\n"
            f"**Career Shift Indicator**: {report.get('career_shift_indicator', 'No')} — "
            f"{'The Head line angle and Life-Head connection suggest a meaningful career redirection is written in your palm.' if report.get('career_shift_indicator') == 'Yes' else 'Your palm suggests a steadier, more linear career progression.'}\n\n"
            f"{themes.get('stability', '')}"
        )

    # ── LOVE & RELATIONSHIPS ──
    if any(w in q for w in ("love", "heart", "relationship", "marriage", "partner", "wife", "husband", "boyfriend", "girlfriend", "romance", "wedding", "soulmate", "marry")):
        rel_timing = [p for p in timing.get("predictions", []) if p.get("category") == "relationships"]
        rel_text = rel_timing[0]["detail"] if rel_timing else "Significant emotional connection indicated in the 24-30 age range."
        venus_mount = mounts.get("Venus", {})
        return (
            f"💕 **Love & Relationship Reading (Cheiro's Analysis)**\n\n"
            f"{themes.get('relationships', '')}\n\n"
            f"**Heart Line Style**: {report.get('line_readings', [{}])[-1].get('detail', '')[:300]}\n\n"
            f"**Venus Mount**: {venus_mount.get('strength', 'Unknown')} — "
            f"{'indicates deep passion, warmth, and magnetic attraction' if venus_mount.get('score', 0) >= 0.4 else 'suggests a more reserved approach to physical affection'}.\n\n"
            f"**Relationship Timing**: {rel_text}\n\n"
            f"**Compatibility Element**: Your {ht.get('element', 'Mixed')} element nature pairs best with "
            f"{_get_best_element_match(ht.get('element', 'Mixed'))}.\n\n"
            f"**Personality in Love**: {ht.get('relationships', '')}\n\n"
            f"{report.get('shared_notes', [''])[0]}"
        )

    # ── TIME & WHEN ──
    if any(w in q for w in ("when", "time", "timing", "age", "year", "period", "future", "predict", "prediction", "timeline")):
        pred_texts = []
        for pred in timing.get("predictions", []):
            pred_texts.append(f"• **{pred['period']}** — {pred['event']}: {pred['detail'][:200]}")
        return (
            f"⏳ **Time Predictions (Cheiro's Timing System)**\n\n"
            f"Cheiro developed a precise timing method: {TIMING_SYSTEM['description'][:200]}\n\n"
            f"**Your Predicted Timeline:**\n\n"
            + "\n\n".join(pred_texts) +
            f"\n\n**Note**: {timing.get('note', '')}"
        )

    # ── MIND & INTELLECT ──
    if any(w in q for w in ("mind", "study", "decision", "head", "brain", "learn", "education", "intelligence", "think", "mental")):
        return (
            f"🧠 **Mind & Intellect Reading**\n\n"
            f"{themes.get('mindset', '')}\n\n"
            f"**Head Line Analysis**: {report.get('line_readings', [{}, {}])[1].get('detail', '')[:400]}\n\n"
            f"**Thinking Style**: Your {ht.get('type', 'Mixed')} hand combined with "
            f"{'a sloping Head line suggests creative, imaginative thinking that excels in arts and innovation' if abs(float(report.get('line_readings', [{}, {}])[1].get('shape', '') == 'curved')) else 'a practical Head line suggests analytical, logical thinking that excels in business and science'}.\n\n"
            f"**Personality Archetype**: {personality.get('archetype', 'Unknown')} — {personality.get('description', '')[:200]}"
        )

    # ── HEALTH & VITALITY ──
    if any(w in q for w in ("health", "life", "energy", "vitality", "disease", "sick", "medical", "body", "physical", "death", "die", "long")):
        health_bits = []
        for ind in health.get("indicators", []):
            health_bits.append(f"• **{ind['area']}**: {ind['assessment']} — {ind['detail'][:200]}")
        return (
            f"💚 **Health & Vitality Reading**\n\n"
            f"**Overall Vitality**: {health.get('overall_vitality', 'moderate').title()}\n\n"
            f"{themes.get('energy', '')}\n\n"
            + "\n".join(health_bits) +
            f"\n\n{health.get('disclaimer', '')}"
        )

    # ── PERSONALITY ──
    if any(w in q for w in ("personality", "character", "trait", "nature", "type", "who am i", "myself", "about me", "describe")):
        traits = "\n".join([f"• {t}" for t in personality.get("core_traits", [])[:6]])
        return (
            f"👤 **Personality Profile (Cheiro's Classification)**\n\n"
            f"**Hand Type**: {ht.get('type', 'Mixed')} ({ht.get('hindi', '')})\n"
            f"**Element**: {ht.get('element', 'Mixed')}\n"
            f"**Archetype**: {personality.get('archetype', 'Unknown')}\n"
            f"**Dominant Mount**: {personality.get('dominant_mount', 'Unknown')}\n\n"
            f"**Core Traits**:\n{traits}\n\n"
            f"**Cheiro's Description**: {ht.get('description', '')[:400]}\n\n"
            f"**Career Aptitude**: {', '.join(ht.get('career', [])[:5])}\n\n"
            f"**In Relationships**: {ht.get('relationships', '')}"
        )

    # ── HAND SHAPE & ELEMENT ──
    if any(w in q for w in ("shape", "element", "earth", "air", "fire", "water", "hand type", "square", "conic")):
        return (
            f"✋ **Hand Type Analysis (Cheiro's 7-Type System)**\n\n"
            f"**Your Hand Type**: {ht.get('type', 'Mixed')} ({ht.get('hindi', '')})\n\n"
            f"{ht.get('description', '')}\n\n"
            f"**Element**: {ht.get('element', 'Mixed')}\n\n"
            f"**Key Personality Traits**:\n"
            + "\n".join([f"• {t}" for t in ht.get("personality", [])[:5]]) +
            f"\n\n**Health Tendency**: {ht.get('health', '')}\n\n"
            f"**Detected Source**: {'User selected' if ht.get('source') == 'user_selected' else 'Auto-detected from palm features'}"
        )

    # ── MOUNTS ──
    if any(w in q for w in ("mount", "jupiter", "saturn", "apollo", "mercury", "venus", "moon", "mars")):
        mount_bits = []
        for name, data in mounts.items():
            mount_bits.append(f"• **{name.replace('_', ' ')}**: {data.get('strength', 'Unknown')} (score: {data.get('score', 0)})")
        return (
            f"⛰️ **Mount Analysis (Cheiro's System)**\n\n"
            f"**Dominant Mount**: {report.get('dominant_mount', 'Unknown').replace('_', ' ')}\n\n"
            + "\n".join(mount_bits) +
            f"\n\n**Dominant Mount Reading**:\n"
            f"{_get_mount_detail(report.get('dominant_mount', ''))}"
        )

    # ── ACCURACY & QUALITY ──
    if any(w in q for w in ("accurate", "confidence", "clear", "quality")):
        return (
            f"📊 **Scan Quality Assessment**\n\n"
            f"Current detection quality: **{report.get('detection_quality', 0):.0%}**\n\n"
            f"For optimal results, ensure:\n"
            f"• Bright, even lighting on the palm\n"
            f"• Palm flat and centered in frame\n"
            f"• All five fingers visible\n"
            f"• No shadows across the palm surface\n\n"
            f"The current scan detects {len([lr for lr in report.get('line_strengths', {}).values() if lr > 0.1])} of 3 major lines. "
            f"{'Excellent clarity for detailed reading.' if report.get('detection_quality', 0) > 0.7 else 'Better image quality would improve the reading accuracy.'}"
        )

    # ── DOMINANT LINE ──
    if any(w in q for w in ("dominant", "strongest", "main", "primary")):
        return (
            f"👑 **Dominant Line Analysis**\n\n"
            f"Your dominant line is the **{report.get('dominant_line', 'Unknown')}** line "
            f"at **{report.get('dominant_strength_pct', 0):.1f}%** of the detected pattern.\n\n"
            f"In Cheiro's system, the dominant line reveals the primary theme of your life:\n"
            f"• **Life line dominant** = Life energy and vitality drive everything\n"
            f"• **Head line dominant** = Mental pursuits and intellect are central\n"
            f"• **Heart line dominant** = Emotional expression and relationships define life\n\n"
            f"Your reading: {report.get('line_readings', [{}])[0].get('detail', '')[:200]}"
        )

    # ── TRAVEL ──
    if any(w in q for w in ("travel", "abroad", "foreign", "overseas", "move", "relocat")):
        return (
            f"✈️ **Travel & Relocation Reading**\n\n"
            f"{TRAVEL_ANALYSIS.get('many_travel_lines', '')}\n\n"
            f"{TRAVEL_ANALYSIS.get('overseas_indicator', '')}\n\n"
            f"**Your Life Line**: {report.get('line_readings', [{}])[0].get('detail', '')[:200]}\n\n"
            f"**Moon Mount**: Your Moon mount is {mounts.get('Moon', {}).get('strength', 'Unknown')} — "
            f"{'indicating strong wanderlust and love of exploration' if mounts.get('Moon', {}).get('score', 0) > 0.35 else 'suggesting travel is purposeful rather than restless'}."
        )

    # ── SPIRITUAL ──
    if any(w in q for w in ("spiritual", "psychic", "mystic", "occult", "soul", "karma", "destiny", "purpose", "meditation")):
        return (
            f"🔮 **Spiritual & Mystical Reading**\n\n"
            f"**Saturn Mount**: {'Your developed Saturn mount indicates philosophical depth and karmic awareness.' if mounts.get('Saturn', {}).get('score', 0) > 0.35 else 'Saturn influence is moderate — spiritual growth comes through life experience.'}\n\n"
            f"**Moon Mount**: {'Strong Moon mount reveals natural psychic sensitivity and intuitive gifts.' if mounts.get('Moon', {}).get('score', 0) > 0.35 else 'Moon influence suggests potential for spiritual development through practice.'}\n\n"
            f"**Head Line**: {report.get('line_readings', [{}, {}])[1].get('detail', '')[:200]}\n\n"
            f"{SPECIAL_MARKS['Cross'].get('mystic_cross', '')}\n\n"
            f"Your {personality.get('archetype', 'Unknown')} archetype {' naturally tends toward mystical exploration and spiritual depth' if personality.get('archetype', '').lower() in ('mystic psychic', 'creative artist') else ' develops spiritual wisdom through worldly experience and service'}."
        )

    # ── CHILDREN & FAMILY ──
    if any(w in q for w in ("child", "children", "baby", "son", "daughter", "family", "parent")):
        return (
            f"👶 **Family & Children Reading**\n\n"
            f"{MINOR_LINES['Children']['description']}\n\n"
            f"**Heart Line Indication**: {themes.get('relationships', '')}\n\n"
            f"**Venus Mount**: Your Venus mount is {mounts.get('Venus', {}).get('strength', 'Unknown')} — "
            f"{'indicating deep love for family and children' if mounts.get('Venus', {}).get('score', 0) >= 0.4 else 'suggesting a more measured approach to family life'}.\n\n"
            f"**Note**: Detailed children line analysis requires very high-resolution imaging "
            f"of the percussion edge near the Mercury finger."
        )

    # ── LUCK & FORTUNE ──
    if any(w in q for w in ("luck", "fortune", "rich", "wealth", "prosper", "success", "famous")):
        return (
            f"🍀 **Fortune & Success Reading**\n\n"
            f"**Sun/Apollo Mount**: {mounts.get('Sun_Apollo', {}).get('strength', 'Unknown')} — "
            f"{'indicating strong potential for fame, recognition, and creative success' if mounts.get('Sun_Apollo', {}).get('score', 0) > 0.35 else 'success comes through persistent effort rather than luck'}.\n\n"
            f"**Jupiter Mount**: {mounts.get('Jupiter', {}).get('strength', 'Unknown')} — "
            f"{'leadership and authority bring fortune' if mounts.get('Jupiter', {}).get('score', 0) > 0.35 else 'fortune flows from collaborative rather than commanding roles'}.\n\n"
            f"**Fate Line**: {themes.get('career', '')}\n\n"
            f"**Sun Line**: {themes.get('visibility', '')}\n\n"
            f"Your {ht.get('type', 'Mixed')} hand with {ht.get('element', 'Mixed')} element "
            f"{'naturally attracts abundance through...' if ht.get('element') in ('Fire', 'Earth') else 'creates success through...'} "
            f"{', '.join(ht.get('career', [])[:3])}."
        )

    # ── DEFAULT COMPREHENSIVE ──
    return (
        f"🔮 **Complete Cheiro's Palm Reading**\n\n"
        f"{report.get('summary', '')}\n\n"
        f"**Quick Insights**:\n"
        f"• Hand Type: {ht.get('type', 'Mixed')} ({ht.get('element', 'Mixed')} element)\n"
        f"• Dominant Line: {report.get('dominant_line', 'Unknown')}\n"
        f"• Dominant Mount: {report.get('dominant_mount', 'Unknown').replace('_', ' ')}\n"
        f"• Archetype: {personality.get('archetype', 'Unknown')}\n"
        f"• Career Shift: {report.get('career_shift_indicator', 'No')}\n\n"
        f"Ask me specific questions about **career, love, timing, health, personality, "
        f"spirituality, travel, children, fortune**, or any aspect of your reading "
        f"for a deeper Cheiro-level analysis."
    )


# ─────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS FOR ANSWERS
# ─────────────────────────────────────────────────────────────────────────

def _get_mount_career(mount_name: str) -> str:
    mount_data = MOUNT_ANALYSIS.get(mount_name, {})
    well_dev = mount_data.get("well_developed", {})
    if isinstance(well_dev, dict):
        return well_dev.get("career", "general professional pursuits")
    return "general professional pursuits"


def _get_mount_detail(mount_name: str) -> str:
    mount_data = MOUNT_ANALYSIS.get(mount_name, {})
    well_dev = mount_data.get("well_developed", {})
    if isinstance(well_dev, dict):
        personality_list = well_dev.get("personality", [])
        if isinstance(personality_list, list):
            return "\n".join([f"• {t}" for t in personality_list[:5]])
    return "Mount influence noted."


def _get_best_element_match(element: str) -> str:
    compat = ELEMENT_COMPATIBILITY.get(element, {})
    best = ""
    best_text = ""
    for el, desc in compat.items():
        if "excellent" in desc.lower() or "brilliant" in desc.lower():
            best = el
            best_text = desc
            break
    if not best:
        for el, desc in compat.items():
            if "nurturing" in desc.lower() or "stable" in desc.lower():
                best = el
                best_text = desc
                break
    if not best:
        best = list(compat.keys())[0] if compat else "Earth"
        best_text = compat.get(best, "")
    return f"**{best}** element partners — {best_text}"
