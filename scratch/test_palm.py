"""Quick validation of the palm module stack."""
from utils.palmistry_knowledge import (
    build_professional_system_prompt,
    get_professional_greeting,
    HAND_TYPES, MOUNT_ANALYSIS, LINE_COMPREHENSIVE,
    TIMING_SYSTEM, HEALTH_INDICATORS, PERSONALITY_PROFILES,
)
from utils.palmistry_engine import (
    build_palm_report, answer_palm_question,
    classify_hand_type, analyze_mounts, predict_timing,
    analyze_health, build_personality_profile,
)

print("✓ Knowledge base imported OK")
print(f"  Hand types: {len(HAND_TYPES)}")
print(f"  Mounts: {len(MOUNT_ANALYSIS)}")
print(f"  Lines: {len(LINE_COMPREHENSIVE)}")
print(f"  Profiles: {len(PERSONALITY_PROFILES)}")

features = {
    'life_length': 200, 'head_length': 180, 'heart_length': 170,
    'life_curvature': 1.15, 'head_curvature': 1.12, 'heart_curvature': 1.18,
    'head_angle': 8, 'life_angle': 5, 'heart_angle': 3,
    'life_head_intersection': 1, 'head_heart_intersection': 0,
    'life_heart_intersection': 0,
}

print("\n✓ Building full report...")
r = build_palm_report(features)
print(f"  Report keys: {list(r.keys())[:12]}")
print(f"  Hand type: {r['hand_type']['type']}")
print(f"  Element: {r['hand_type']['element']}")
print(f"  Archetype: {r['personality']['archetype']}")
print(f"  Dominant line: {r['dominant_line']}")
print(f"  Dominant mount: {r['dominant_mount']}")
print(f"  Timing predictions: {len(r['timing']['predictions'])}")
print(f"  Health indicators: {len(r['health']['indicators'])}")
print(f"  Mount scores: {list(r['mounts'].keys())}")
print(f"  Chat context length: {len(r['chat_context'])} chars")

print("\n✓ Testing answer_palm_question...")
test_questions = [
    "When will I get married?",
    "What about my career?",
    "Tell me about my personality",
    "What time period shows major changes?",
    "Am I healthy?",
    "What is my hand type?",
    "What do my mounts say?",
    "Will I travel abroad?",
]
for q in test_questions:
    a = answer_palm_question(q, r)
    print(f"  Q: {q[:40]:40s} → {len(a):4d} chars")

print("\n✓ Testing professional prompt...")
prompt = build_professional_system_prompt()
print(f"  System prompt length: {len(prompt)} chars")
greeting = get_professional_greeting()
print(f"  Greeting length: {len(greeting)} chars")

print("\n═══ ALL CHECKS PASSED ═══")
