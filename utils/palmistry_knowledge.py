"""
══════════════════════════════════════════════════════════════════════════════
CHEIRO'S PALMISTRY — COMPLETE KNOWLEDGE BASE
══════════════════════════════════════════════════════════════════════════════
Derived from *Cheiro Ki Hastrekhayein* (कीरो की हस्तरेखाएँ) — the Hindi
translation of Count Louis Hamon's (Cheiro) authoritative palmistry system.
204 pages covering 24 chapters across 3 parts.

This module encodes every detail — hand types, mounts, lines, marks,
timing, health indicators, and personality profiling — as structured
Python data for use by the palmistry engine and AI chatbot.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — HAND TYPES (Ch. 1–8)
# Cheiro classifies hands into 7 primary categories
# ═══════════════════════════════════════════════════════════════════════════

HAND_TYPES: Dict[str, Dict[str, Any]] = {
    "Elementary": {
        "hindi": "अविकसित या निम्न श्रेणी का हाथ",
        "description": (
            "The Elementary hand has a broad, thick, heavy palm with short, "
            "clumsy fingers and very few lines. The thumb is short and stiff. "
            "This is the most primitive hand type according to Cheiro."
        ),
        "personality": [
            "Governed by brute force and basic instincts",
            "Little intellectual development or imagination",
            "Passionate but lacks control over emotions",
            "Violent temper when provoked, but generally slow in thought",
            "Superstitious nature, easily influenced by environment",
            "Prefers physical labor over mental work",
        ],
        "career": ["Manual labor", "Agriculture", "Mining", "Physical trades"],
        "health": "Robust physical health but prone to excesses. Risk of injuries from physical work.",
        "relationships": "Basic emotional expression. Loyal once attached but lacks romantic finesse.",
        "detection_hints": {
            "palm_width_ratio": "> 1.2",  # wider than tall
            "finger_length": "short",
            "line_count": "very few (3-4 major only)",
            "thumb_length": "short",
        },
    },
    "Square": {
        "hindi": "वर्गाकार हाथ",
        "description": (
            "The Square hand has a square-shaped palm, square fingertips, "
            "and a generally robust appearance. The wrist, base of fingers, "
            "and overall outline form a rectangular shape. Cheiro identifies "
            "8 sub-types based on finger characteristics."
        ),
        "sub_types": {
            "short_square_fingers": "Methodical, orderly, practical. Loves routine and system.",
            "long_square_fingers": "More intellectual but still practical. Analytical mind.",
            "knotty_square_fingers": "Philosophical, scientific temperament. Seeks proof before belief.",
            "smooth_square_fingers": "Intuitive yet practical. Quick learner, artistic leanings.",
            "spatulate_square_fingers": "Action-oriented, inventive. Loves mechanics and engineering.",
            "conical_square_fingers": "Artistic with practical application. Music, applied arts.",
            "psychic_square_fingers": "Rare combination. Visionary ideas with ability to execute.",
            "mixed_square_fingers": "Versatile, adaptable. Jack of all trades, master of some.",
        },
        "personality": [
            "Methodical, orderly, and systematic in all things",
            "Loves routine, discipline, and established rules",
            "Persevering and determined — never gives up easily",
            "Practical rather than imaginative",
            "Respects law, order, and social conventions",
            "Reliable, punctual, and honest in dealings",
            "Can be rigid and stubborn when challenged",
            "Rational thinker who demands proof and evidence",
        ],
        "career": [
            "Engineering", "Architecture", "Law", "Business management",
            "Military", "Civil service", "Accounting", "Teaching",
            "Agriculture", "Real estate",
        ],
        "health": "Generally good health due to disciplined lifestyle. May suffer from digestive issues due to regularity obsession.",
        "relationships": "Steady, faithful partner. May lack romantic spontaneity but provides security and stability.",
        "detection_hints": {
            "palm_shape": "rectangular/square",
            "fingertip_shape": "square/flat",
            "palm_width_ratio": "≈ 1.0",
        },
    },
    "Spatulate": {
        "hindi": "चमचाकार अथवा चपटा हाथ",
        "description": (
            "The Spatulate hand has fan-shaped or spatula-like fingertips "
            "that are wider at the tip than at the base. The palm may be "
            "broad at the wrist or at the base of fingers. This indicates "
            "an active, energetic, and inventive nature."
        ),
        "personality": [
            "Extremely energetic, restless, always in action",
            "Independent thinker, hates following conventions",
            "Inventive and original — constantly innovating",
            "Loves travel, exploration, and adventure",
            "Practical genius — takes ideas and makes them work",
            "Brave, courageous, and willing to take risks",
            "Self-confident, sometimes to the point of arrogance",
            "Impatient with slow or methodical people",
        ],
        "career": [
            "Inventor", "Explorer", "Engineer", "Entrepreneur",
            "Pilot", "Surgeon", "Sports professional", "Mechanic",
            "Researcher", "Technology",
        ],
        "health": "High energy but burns out. Needs physical activity. Risk of accidents from recklessness.",
        "relationships": "Exciting but demanding partner. Needs space and independence. Gets bored easily.",
        "detection_hints": {
            "fingertip_shape": "wider at tips",
            "energy_level": "high",
        },
    },
    "Philosophic": {
        "hindi": "दार्शनिक हाथ",
        "description": (
            "The Philosophic hand is long and angular with bony fingers "
            "that have developed joints (knots). The fingertips are "
            "half-square, half-conical. Nails are long. This hand belongs "
            "to deep thinkers, scholars, and spiritual seekers."
        ),
        "personality": [
            "Deep thinker, philosopher, and analyst",
            "Seeks truth in everything, questions established beliefs",
            "Silent, reserved, secretive about personal matters",
            "Studies both religion and science with equal intensity",
            "Proud, dignified, but can be aloof and detached",
            "Dislikes crowds, prefers solitude and study",
            "Mystical inclinations, drawn to occult and metaphysics",
            "Excellent teacher and writer",
        ],
        "career": [
            "Philosophy", "Teaching", "Writing", "Theology",
            "Research", "Psychology", "Diplomacy", "Spiritual guidance",
            "Law", "Counseling",
        ],
        "health": "Tends toward nervous ailments and mental exhaustion. Must guard against melancholy.",
        "relationships": "Selective and deep connections. Few friends but fiercely loyal. Values intellectual compatibility.",
        "detection_hints": {
            "finger_joints": "knotty/prominent",
            "finger_length": "long",
            "palm_shape": "long, narrow",
        },
    },
    "Conic": {
        "hindi": "शंकु आकार का हाथ",
        "description": (
            "The Conic hand (also called Artistic hand) has a slightly "
            "tapering palm with smooth, conical fingertips. The fingers "
            "are medium length and the overall hand has a graceful "
            "appearance. This is the hand of artists and creatives."
        ),
        "personality": [
            "Artistic, creative, and aesthetically sensitive",
            "Impulsive and guided by instinct rather than reason",
            "Loves beauty, luxury, and comfort",
            "Emotionally responsive, empathetic",
            "Changeable moods, easily influenced by surroundings",
            "Generous and warm-hearted but can be extravagant",
            "Quick thinker but may lack follow-through",
            "Loves music, art, poetry, and nature",
        ],
        "career": [
            "Artist", "Musician", "Writer", "Interior designer",
            "Fashion", "Acting", "Photography", "Hospitality",
            "Counselor", "Public relations",
        ],
        "health": "Sensitive constitution. Prone to nervous conditions, allergies, and stress-related ailments.",
        "relationships": "Romantic, passionate, and affectionate. Falls in and out of love quickly. Needs emotional stimulation.",
        "detection_hints": {
            "fingertip_shape": "tapered/conical",
            "finger_surface": "smooth",
        },
    },
    "Psychic": {
        "hindi": "अत्यन्त नुकीला हाथ",
        "description": (
            "The Psychic hand is the most beautiful and rarest of all "
            "hand types. It has a long, slender, delicate palm with very "
            "long, tapering fingers that end in pointed/almond tips. "
            "Cheiro considers this the hand of idealists and visionaries."
        ),
        "personality": [
            "Extreme idealist, dreamer, and visionary",
            "Highly intuitive, almost psychic in perception",
            "Deeply spiritual, drawn to mysticism and the occult",
            "Gentle, trusting, and easily deceived by others",
            "Lives in an inner world of imagination and beauty",
            "Impractical in worldly matters, poor with money",
            "Sensitive to atmosphere and emotions of others",
            "Can be deeply religious or spiritually evolved",
        ],
        "career": [
            "Spiritual teacher", "Healer", "Poet", "Mystic",
            "Fine artist", "Religion", "Charity work",
            "Psychic/Intuitive counselor",
        ],
        "health": "Fragile health, highly sensitive to environment. Prone to nervous disorders and psychosomatic illness.",
        "relationships": "Idealistic in love. Often disappointed by reality. Needs a protective, understanding partner.",
        "detection_hints": {
            "fingertip_shape": "pointed/almond",
            "finger_length": "very long",
            "palm_shape": "long, slender",
        },
    },
    "Mixed": {
        "hindi": "मिश्रित लक्षणों वाला हाथ",
        "description": (
            "The Mixed hand combines features from two or more hand types. "
            "It is by far the most common type found in practice. The "
            "key to reading this hand is identifying which type dominates "
            "and which secondary influences are present."
        ),
        "personality": [
            "Versatile and adaptable — can adjust to many situations",
            "Multiple talents but may lack depth in any single one",
            "Changeable interests and goals over a lifetime",
            "Generally quick-witted and good conversationalist",
            "Can see multiple perspectives on any issue",
            "May struggle with commitment and focus",
            "Influenced by the dominant hand type sub-traits",
        ],
        "career": [
            "Depends on dominant type", "Journalism", "Sales",
            "Consulting", "Multi-career path", "General business",
        ],
        "health": "Varies by dominant type. Generally moderate health with periods of stress.",
        "relationships": "Adaptable partner. Can harmonize with many personality types but needs variety.",
        "detection_hints": {
            "mixed_features": True,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# THUMB ANALYSIS (Ch. 9)
# ═══════════════════════════════════════════════════════════════════════════

THUMB_ANALYSIS: Dict[str, Dict[str, str]] = {
    "flexibility": {
        "very_flexible": (
            "A thumb that bends far back (supple/flexible) indicates a generous, "
            "extravagant, and adaptable nature. Such persons are open-minded, "
            "tolerant, and spend money freely. They adapt easily to new situations "
            "but may lack persistence."
        ),
        "moderately_flexible": (
            "A moderately flexible thumb shows balance between generosity and "
            "prudence. The person is adaptable yet principled, reasonable in "
            "expenditure, and balanced in character."
        ),
        "stiff": (
            "A stiff thumb that does not bend back indicates a stubborn, "
            "determined, and economical nature. Such persons are extremely "
            "persistent, cautious with money, and strong-willed. They resist "
            "change and hold firm to their opinions."
        ),
    },
    "phalanx_ratio": {
        "will_dominant": (
            "When the first phalanx (nail phalanx) of the thumb is longer than "
            "the second, WILLPOWER dominates over logic. The person acts on "
            "determination and force of will rather than reason. They can be "
            "tyrannical if the thumb is also thick and clubbed."
        ),
        "logic_dominant": (
            "When the second phalanx (middle) is longer than the first, LOGIC "
            "dominates over will. The person reasons everything out before acting. "
            "They may lack the drive to execute despite having brilliant plans."
        ),
        "balanced": (
            "When both phalanges are roughly equal, will and logic are balanced. "
            "This is the ideal configuration — the person thinks clearly AND "
            "acts decisively. Most successful people have this ratio."
        ),
    },
    "size": {
        "large": (
            "A large, well-formed thumb indicates strong personality, leadership "
            "ability, and capacity for great achievement. Historically all great "
            "leaders, generals, and rulers had large thumbs."
        ),
        "small": (
            "A small thumb indicates a person ruled by heart rather than head. "
            "They are sentimental, yielding, and easily influenced. They follow "
            "rather than lead and often depend on others for decisions."
        ),
        "very_small": (
            "A very small, weak thumb shows lack of will and determination. "
            "The person drifts through life without strong purpose. They may "
            "be talented but fail to capitalize on opportunities."
        ),
    },
    "shape": {
        "clubbed": (
            "A clubbed (bulbous/murderer's) thumb — where the nail phalanx is "
            "thick and club-shaped — indicates potential for explosive, "
            "uncontrollable rage. In extreme cases, Cheiro associated this "
            "with dangerous criminal tendencies when other signs confirm."
        ),
        "waisted": (
            "A waisted (narrow at the joint) thumb indicates tactful diplomacy "
            "and refined expression. The person can convince others through "
            "charm and persuasion rather than force."
        ),
        "straight": (
            "A straight, well-proportioned thumb indicates honesty, directness, "
            "and straightforward dealing. The person says what they mean and "
            "has no hidden agendas."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FINGER ANALYSIS (Ch. 10–11)
# ═══════════════════════════════════════════════════════════════════════════

FINGER_MEANINGS: Dict[str, Dict[str, str]] = {
    "Jupiter (Index)": {
        "long": "Ambition, leadership desire, pride, love of power and authority. Born to command.",
        "short": "Shy, lacks self-confidence, prefers following to leading. May hide great talent.",
        "normal": "Healthy ambition balanced with humility. Natural leader when needed.",
        "dominant": "The person craves recognition, authority, and social status above all.",
    },
    "Saturn (Middle)": {
        "long": "Serious, melancholic, loves solitude. Deep thinker, philosophical, prudent with money.",
        "short": "Frivolous, superficial, avoids responsibility. Lives for the moment.",
        "normal": "Balance between seriousness and enjoyment. Practical wisdom.",
        "dominant": "Life ruled by fate, destiny, and karmic patterns. Strong sense of duty.",
    },
    "Apollo/Sun (Ring)": {
        "long": "Artistic temperament, love of beauty, risk-taking in speculation. Creative genius.",
        "short": "Practical, dislikes art and beauty. May have undeveloped creative side.",
        "normal": "Appreciates art and beauty without being consumed by it. Balanced creativity.",
        "dominant": "Life revolves around art, fame, and creative expression. Born performer.",
    },
    "Mercury (Little)": {
        "long": "Eloquent speaker, shrewd businessperson, scientific mind. Gift of persuasion.",
        "short": "Difficulty in self-expression. Struggles to communicate ideas effectively.",
        "normal": "Adequate communication skills. Balance between speaking and listening.",
        "dominant": "Born communicator, salesperson, or diplomat. Can convince anyone of anything.",
    },
}

FINGER_JOINTS: Dict[str, str] = {
    "smooth_joints": (
        "Smooth joints (no visible knots) indicate an intuitive, impulsive nature. "
        "The person grasps ideas quickly through inspiration rather than analysis. "
        "They are artistic, creative, and hate tedious detail work."
    ),
    "knotty_first_joint": (
        "A developed first (philosophical) knot indicates methodical thinking and "
        "a love of order in the mental realm. The person analyzes everything "
        "and needs to understand the 'why' before accepting anything."
    ),
    "knotty_second_joint": (
        "A developed second (material) knot indicates orderliness in material life. "
        "The person is neat, systematic, punctual, and organized in practical matters."
    ),
    "both_knotty": (
        "Both joints developed indicates a deeply analytical, systematic person "
        "who needs order in both thought and material life. These are scientists, "
        "researchers, and scholars who leave nothing to chance."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# NAILS ANALYSIS (Ch. 13)
# ═══════════════════════════════════════════════════════════════════════════

NAIL_ANALYSIS: Dict[str, Dict[str, str]] = {
    "shape": {
        "long_narrow": "Gentle, refined temperament. Prone to chest/respiratory ailments. Idealistic nature.",
        "short_broad": "Critical, argumentative nature. Strong constitution. Quick temper, sharp tongue.",
        "very_short": "Quarrelsome, impatient, and fault-finding. Heart and circulation issues possible.",
        "almond": "Peaceful, diplomatic nature. Good health generally. Creative and artistic.",
        "square": "Practical, balanced temperament. Good health. Orderly mind.",
        "fan_shaped": "Nervous temperament, highly strung. Sensitive to stress. Original thinker.",
    },
    "color": {
        "pink": "Good circulation, healthy constitution. Normal temperament.",
        "white": "Cold, selfish nature. Possible anemia or poor circulation.",
        "red": "Aggressive, passionate nature. Strong blood, hot temper. Heart disease risk.",
        "bluish": "Poor circulation, heart/lung issues. Melancholic temperament.",
        "yellow": "Liver/bile issues. Irritable temperament. Tendency toward pessimism.",
    },
    "texture": {
        "ridged": "Nervous tension, stress, nutritional deficiency. Current health challenges.",
        "smooth": "Good health, balanced constitution. Calm nervous system.",
        "brittle": "Mineral deficiency, thyroid issues. Anxious temperament.",
        "spotted_white": "Nervous exhaustion, zinc deficiency. Period of stress.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MOUNTS (Multiple chapters)
# ═══════════════════════════════════════════════════════════════════════════

MOUNT_ANALYSIS: Dict[str, Dict[str, Any]] = {
    "Jupiter": {
        "location": "Below the index finger",
        "planet": "Jupiter (बृहस्पति)",
        "well_developed": {
            "personality": [
                "Natural leader, ambitious and commanding",
                "Deeply religious or philosophical",
                "Love of honor, dignity, and social distinction",
                "Generous, magnanimous, and warm-hearted",
                "Fond of ceremony, ritual, and pageantry",
                "Excellent judge of character",
                "Commands respect naturally without demanding it",
            ],
            "career": "Leadership, administration, religion, law, politics, teaching",
            "love": "Idealistic in love. Seeks a worthy partner of equal status. Devoted once committed.",
        },
        "under_developed": {
            "personality": "Lacks ambition, self-confidence, and desire for achievement.",
            "impact": "May have talent but lacks the drive to use it. Self-doubt dominates.",
        },
        "over_developed": {
            "personality": "Excessive pride, arrogance, tyranny. Domineering nature.",
            "impact": "Crushes others with ego. May become a despot or religious fanatic.",
        },
        "marks": {
            "star": "Brilliant, sudden rise to power and distinction. Great honors befalling the person unexpectedly.",
            "cross": "Happy marriage or union. Love match that elevates social status.",
            "square": "Protection from excess ambition. Guards against fall from power.",
            "triangle": "Exceptional diplomatic skill. Political genius.",
            "circle": "Fame and fortune. Extraordinary success in chosen field.",
            "grid": "Exaggerated ego, failed ambitions, disappointment in power struggles.",
            "island": "Weakened ambition. Loss of respect or position.",
            "dot": "Sudden blow to reputation or position.",
        },
    },
    "Saturn": {
        "location": "Below the middle finger",
        "planet": "Saturn (शनि)",
        "well_developed": {
            "personality": [
                "Serious, prudent, and deeply thoughtful",
                "Love of solitude and independent work",
                "Strong sense of duty and responsibility",
                "Cautious with money, excellent financial planner",
                "Drawn to agriculture, mining, and earth-related work",
                "Melancholic temperament, prone to depression",
                "Philosophical about life and death",
            ],
            "career": "Mining, agriculture, real estate, science, research, occult studies, theology",
            "love": "Fatalistic about love. May marry late or experience loss in love. Devoted but gloomy.",
        },
        "under_developed": {
            "personality": "Insignificant, ordinary life. No special destiny or karmic weight.",
            "impact": "Life flows without major highs or lows. Uneventful existence.",
        },
        "over_developed": {
            "personality": "Extreme melancholy, morbid thoughts, potential for self-harm.",
            "impact": "Dark outlook on life. Isolation, cynicism, and deep pessimism.",
        },
        "marks": {
            "star": "Fatality or great danger. Could indicate sudden dramatic life event.",
            "cross": "Tendency toward mysticism and occult. May foretell danger from accidents.",
            "square": "Protection from accidents and fatalistic events. A guardian sign.",
            "triangle": "Aptitude for occult sciences, deep metaphysical understanding.",
            "grid": "Misfortune, losses, and a life of continuous struggle.",
            "island": "Financial losses. Weakened destiny pattern.",
        },
    },
    "Sun_Apollo": {
        "location": "Below the ring finger",
        "planet": "Sun/Apollo (सूर्य)",
        "well_developed": {
            "personality": [
                "Brilliant, creative, and artistically gifted",
                "Love of beauty in all forms — art, music, literature",
                "Versatile genius with multiple talents",
                "Sunny disposition, optimistic, and charming",
                "Desire for fame and public recognition",
                "Generous, warm, and inspires loyalty in others",
                "Natural sense of style and aesthetics",
            ],
            "career": "Art, music, acting, literature, design, public speaking, media, luxury goods",
            "love": "Passionate, demonstrative lover. Attracted to beauty and glamour. May be vain.",
        },
        "under_developed": {
            "personality": "Dull existence without art or beauty. Philistine nature.",
            "impact": "No desire for fame or recognition. Content with mediocrity.",
        },
        "over_developed": {
            "personality": "Vanity, extravagance, love of display. Shallow pursuit of fame.",
            "impact": "Sacrifices substance for appearance. All show and no depth.",
        },
        "marks": {
            "star": "Tremendous fame and fortune through art or talent. Celebrity status.",
            "cross": "Failure in artistic pursuits despite talent. Blocked creativity.",
            "square": "Protection of reputation and artistic legacy.",
            "triangle": "Combining art with practical skill. Commercial success in creativity.",
            "grid": "Vanity and desire for fame without talent to back it up.",
            "island": "Loss of reputation. Scandal affecting public image.",
        },
    },
    "Mercury": {
        "location": "Below the little finger",
        "planet": "Mercury (बुध)",
        "well_developed": {
            "personality": [
                "Quick-witted, eloquent, and persuasive speaker",
                "Sharp business acumen and commercial instinct",
                "Scientific mind combined with communication skill",
                "Love of travel for business and knowledge",
                "Resourceful, adaptable, and mentally agile",
                "Excellent with languages and writing",
                "Natural diplomat and negotiator",
            ],
            "career": "Business, commerce, medicine, science, law, writing, diplomacy, communication",
            "love": "Intellectual attraction first. Values mental compatibility. May be calculating in love.",
        },
        "under_developed": {
            "personality": "Poor communication skills. Difficulty in business and negotiation.",
            "impact": "Struggles to express ideas. May be taken advantage of commercially.",
        },
        "over_developed": {
            "personality": "Cunning, deceitful, and manipulative. Uses speech to deceive.",
            "impact": "Fraud, theft, and dishonesty. Words cannot be trusted.",
        },
        "marks": {
            "star": "Exceptional success in business, science, or oratory. Brilliant speaker or scientist.",
            "cross": "Tendency toward dishonesty. Danger of being caught in deception.",
            "square": "Protection in business dealings. Guards against financial loss.",
            "triangle": "Diplomatic genius. Success in politics or international affairs.",
            "grid": "Dishonesty as a way of life. Habitual deceiver.",
        },
    },
    "Mars_Upper": {
        "location": "Below Mercury mount, on the percussion side",
        "planet": "Mars (मंगल) — Upper/Passive",
        "well_developed": {
            "personality": [
                "Moral courage and endurance under pressure",
                "Calm resistance and passive bravery",
                "Ability to bear suffering without complaint",
                "Self-control and inner strength",
                "Persistence and determination in the face of obstacles",
                "Controls temper through willpower",
            ],
            "career": "Military (strategic roles), endurance sports, crisis management, counseling",
            "love": "Patient and enduring in relationships. Stays through difficult times.",
        },
        "over_developed": {
            "personality": "Violent temper, cruelty, and aggression.",
            "impact": "Dangerous when provoked. Physical confrontation.",
        },
    },
    "Mars_Lower": {
        "location": "Above the thumb, between Jupiter and Venus mounts",
        "planet": "Mars (मंगल) — Lower/Active",
        "well_developed": {
            "personality": [
                "Physical courage and aggressive bravery",
                "Martial spirit, love of contest and competition",
                "Bold, daring, and acts on impulse",
                "Cannot tolerate injustice or bullying",
                "Quick temper but equally quick to forgive",
                "Thrives in conflict and challenge",
            ],
            "career": "Military, police, firefighting, sports, surgery, emergency services",
            "love": "Passionate, jealous, and possessive. Fights for love.",
        },
    },
    "Moon": {
        "location": "Lower percussion side, below upper Mars",
        "planet": "Moon/Luna (चन्द्र)",
        "well_developed": {
            "personality": [
                "Rich imagination, powerful dreamer",
                "Love of travel, especially sea voyages",
                "Romantic, poetic, and mystical nature",
                "Strong intuition and psychic sensitivity",
                "Restless nature, constantly seeking change",
                "Creative storyteller with vivid inner world",
                "Love of music, especially romantic and atmospheric genres",
            ],
            "career": "Travel industry, navy, writing, music, painting, psychic work, oceanography",
            "love": "Deeply romantic but changeable. Falls in love with love itself. Needs constant novelty.",
        },
        "under_developed": {
            "personality": "Lacking imagination and creativity. Unimaginative, dull nature.",
            "impact": "Cannot appreciate art, music, or beauty. Purely materialistic.",
        },
        "over_developed": {
            "personality": "Excessive imagination, delusions, and fantasy. Disconnected from reality.",
            "impact": "May suffer from depression, hallucinations, or insomnia. Dangerously restless.",
        },
        "marks": {
            "star": "Fame through imagination. May also indicate danger from water.",
            "cross": "Danger of drowning or water-related mishap. False imagination.",
            "triangle": "Great intuitive talent. Prophetic ability.",
            "grid": "Restlessness, discontent, perpetual dissatisfaction.",
        },
    },
    "Venus": {
        "location": "Base of thumb, encircled by the Life line",
        "planet": "Venus (शुक्र)",
        "well_developed": {
            "personality": [
                "Passionate, warm-blooded, and sensual",
                "Deep love of beauty, music, and pleasure",
                "Generous, compassionate, and kind-hearted",
                "Magnetic personality that attracts others",
                "Love of dancing, singing, and celebration",
                "Need for human connection and intimacy",
                "Grace, charm, and social elegance",
            ],
            "career": "Music, dance, fashion, hospitality, beauty industry, social work, matchmaking",
            "love": (
                "Deeply passionate and romantic. Falls in love easily and intensely. "
                "Physical attraction is paramount. May have multiple relationships."
            ),
        },
        "under_developed": {
            "personality": "Cold, passionless, and detached. Lacks warmth and empathy.",
            "impact": "Difficulty forming intimate relationships. Emotionally unavailable.",
        },
        "over_developed": {
            "personality": "Excessive sensuality, vanity, and self-indulgence.",
            "impact": (
                "Cheiro warns: when Venus mount is excessively high and the "
                "Head line is weak, the person is enslaved by passions and "
                "may act on base desires without moral restraint."
            ),
        },
        "marks": {
            "star": "Extraordinary success in love and beauty. Magnetic attraction.",
            "cross": "Significant love affair that profoundly affects life.",
            "triangle": "Mastery over passions. Calculated romantic choices.",
            "grid": "Extreme sensuality and restlessness in love.",
            "island": "Scandal or disgrace through love affairs.",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MAJOR LINES — COMPREHENSIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

LINE_COMPREHENSIVE: Dict[str, Dict[str, Any]] = {
    "Life": {
        "hindi": "जीवन रेखा",
        "location": "Curves around the Venus mount from between Jupiter and thumb to the base of palm",
        "governs": [
            "Quality and vitality of life (NOT length of life)",
            "Physical constitution and stamina",
            "Major life changes, transitions, and turning points",
            "Travel and relocation events",
            "Energy levels throughout different life periods",
            "Influence of family and home environment",
        ],
        "variations": {
            "long_well_marked": (
                "A long, deeply marked Life line without breaks indicates robust "
                "health, strong vitality, and a vigorous life. The person has "
                "tremendous physical energy and recovers quickly from illness."
            ),
            "short": (
                "Cheiro clarifies: a short Life line does NOT necessarily mean "
                "a short life. It may indicate less physical vitality, more "
                "concentrated energy, or a life that is intense rather than long. "
                "Always check the Fate line as supplement."
            ),
            "chained": (
                "A chained Life line indicates continuous health problems, "
                "especially in early life. Delicate constitution, frequent "
                "illness, and lack of robust energy."
            ),
            "double": (
                "A double Life line (sister line running parallel inside) is "
                "an extremely favorable sign. It indicates extra vitality, "
                "protection from danger, and often military or government "
                "service. Some traditions say it indicates support from a "
                "very close person throughout life."
            ),
            "broken": (
                "A broken Life line indicates a serious illness, accident, "
                "or major life change at the age where the break occurs. "
                "If the line resumes strongly after the break, the person "
                "recovers and continues with renewed energy."
            ),
            "forked_at_end": (
                "A fork at the end of the Life line indicates travel or "
                "relocation in later years. The larger fork often points "
                "toward the direction of travel (toward Moon mount = "
                "overseas travel)."
            ),
            "branches_upward": (
                "Upward branches from the Life line indicate periods of "
                "success, achievement, and elevation. Each branch represents "
                "a rise in fortune at that time period."
            ),
            "branches_downward": (
                "Downward branches indicate periods of loss, fatigue, or "
                "decline. Financial or health setbacks at those periods."
            ),
            "starting_from_jupiter": (
                "A Life line starting from the Jupiter mount (high start) "
                "indicates extraordinary ambition and a person who achieves "
                "through sheer willpower and determination."
            ),
            "joined_with_head": (
                "When the Life line and Head line are joined at the start, "
                "it indicates a cautious, careful nature in early life. "
                "The longer they stay joined, the more the person is "
                "dependent on family influence before gaining independence."
            ),
            "separated_from_head": (
                "When the Life line and Head line are separated from the "
                "start, it indicates an independent nature from early age. "
                "The person makes their own decisions early in life."
            ),
            "island": (
                "An island on the Life line indicates a period of weakness, "
                "illness, or confinement. The size and position indicate "
                "duration and timing of the health challenge."
            ),
            "cross_bars": (
                "Small lines crossing the Life line indicate obstacles, "
                "worries, and interference from others at those time periods."
            ),
        },
        "timing_method": (
            "Cheiro's timing on the Life line: The line is read from its "
            "start (between Jupiter and thumb) downward toward the wrist. "
            "Divide the line into segments: the point where a line drawn "
            "from the middle of Saturn finger base meets the Life line "
            "marks approximately age 20. The midpoint of the line marks "
            "approximately age 40. Continue proportionally to the base."
        ),
    },
    "Head": {
        "hindi": "शीर्ष/मस्तिष्क रेखा",
        "location": "Runs across the palm from between Jupiter mount and thumb toward the percussion",
        "governs": [
            "Mental capacity and intellectual power",
            "Thinking style — practical vs imaginative",
            "Decision-making ability and approach",
            "Concentration and focus",
            "Learning ability and educational inclinations",
            "Mental health and psychological stability",
        ],
        "variations": {
            "long_straight": (
                "A long, straight Head line indicates a practical, analytical, "
                "and business-oriented mind. The person thinks logically, "
                "makes decisions based on facts, and excels in commerce "
                "and administration."
            ),
            "long_sloping_to_moon": (
                "A Head line that slopes downward to the Moon mount indicates "
                "a rich imagination and creative mind. The steeper the slope, "
                "the more imagination dominates. Artists, writers, and poets "
                "typically have this configuration."
            ),
            "short": (
                "A short Head line indicates a mind focused on physical/material "
                "matters. The person is practical but lacks interest in "
                "intellectual or abstract pursuits."
            ),
            "forked_end": (
                "A fork at the end of the Head line (Writer's Fork) is one of "
                "the most favorable signs. It combines imagination with "
                "practical thinking — perfect for successful writers, "
                "diplomats, and business people with creative vision."
            ),
            "chained": (
                "A chained Head line indicates difficulty concentrating, "
                "headaches, and a mind that is easily disturbed. Lack of "
                "consistent mental focus."
            ),
            "double": (
                "A double Head line indicates exceptional mental power. "
                "The person has two distinct modes of thinking — practical "
                "and creative — and can switch between them at will."
            ),
            "broken": (
                "A break in the Head line indicates a severe mental shock, "
                "head injury, or period of mental instability at the "
                "corresponding age."
            ),
            "island": (
                "An island on the Head line indicates a period of mental "
                "strain, confusion, or brain-related illness. Cheiro noted "
                "this commonly in people who experienced periods of severe "
                "worry or depression."
            ),
            "straight_across": (
                "A Head line that runs perfectly straight across the palm "
                "(Simian-like tendency) indicates extreme single-minded "
                "focus. The person is intensely concentrated but struggles "
                "to see other perspectives."
            ),
            "deep_well_marked": (
                "A deep, well-marked Head line indicates powerful intellect, "
                "strong concentration, and ability to influence others through "
                "mental force."
            ),
            "wavy": (
                "A wavy or wavering Head line indicates inconsistency in "
                "thinking, shifting focus, and difficulty maintaining a "
                "single line of thought."
            ),
        },
        "timing_method": (
            "Timing on the Head line: Read from the start (under Jupiter) "
            "toward the percussion. A vertical line from Saturn finger base "
            "marks approximately age 20-25. A line from Apollo finger base "
            "marks approximately age 35-40."
        ),
    },
    "Heart": {
        "hindi": "हृदय रेखा",
        "location": "Runs across the upper palm from under Mercury toward Jupiter/Saturn",
        "governs": [
            "Emotional nature and expression of feelings",
            "Capacity for love and romantic relationships",
            "Loyalty and devotion in partnerships",
            "Heart health and cardiovascular system",
            "Emotional stability and resilience",
            "Sexual nature and desires",
        ],
        "variations": {
            "ending_under_jupiter": (
                "A Heart line ending under the Jupiter finger indicates "
                "idealistic love. The person loves with the head and heart "
                "combined. They idealize their partner, are fiercely devoted, "
                "and demand the highest standards in love."
            ),
            "ending_on_jupiter_mount": (
                "Heart line ending ON the Jupiter mount indicates intense, "
                "all-consuming love. The person gives everything in love "
                "and can be possessively devoted."
            ),
            "ending_between_jupiter_saturn": (
                "Heart line ending between Jupiter and Saturn is the most "
                "balanced position. The person loves deeply but maintains "
                "reason. Calm, steady, and reliable in relationships."
            ),
            "ending_under_saturn": (
                "Heart line ending under Saturn indicates a more physical, "
                "sensual love nature. The person is practical about "
                "relationships and may lack romantic idealism."
            ),
            "straight": (
                "A straight Heart line indicates a controlled emotional nature. "
                "The person is reasonable in love, thinks before feeling, and "
                "maintains composure even in emotional situations."
            ),
            "curved": (
                "A curved Heart line indicates warmth, expressiveness, and "
                "demonstrative affection. The person shows love openly "
                "and needs physical expressions of affection."
            ),
            "chained": (
                "A chained Heart line indicates emotional sensitivity to the "
                "extreme, weak heart health, and a tendency to be hurt "
                "repeatedly in love. Fickle in affections."
            ),
            "broken": (
                "A broken Heart line indicates heartbreak, loss in love, "
                "or serious emotional trauma at the age of the break. "
                "May also indicate heart disease."
            ),
            "forked_start": (
                "A Heart line with a fork at the start (under Mercury) "
                "indicates high emotion meeting analytical ability. "
                "The person understands their emotions intellectually."
            ),
            "branches_upward": (
                "Upward branches from the Heart line indicate happy "
                "attachments and successful love affairs at those periods."
            ),
            "branches_downward": (
                "Downward branches indicate disappointments, heartbreaks, "
                "and failed relationships at those periods."
            ),
            "absent": (
                "Absence of clear Heart line (or merging with Head line "
                "in a Simian line) indicates a person ruled entirely by "
                "head over heart. Cold in affection but extremely loyal."
            ),
        },
    },
    "Fate_Saturn": {
        "hindi": "भाग्य रेखा",
        "location": "Runs vertically from base of palm toward Saturn (middle) finger",
        "governs": [
            "Career path and professional destiny",
            "Financial success and material achievement",
            "External circumstances affecting life direction",
            "Social status and worldly position",
            "Influence of fate versus free will in career",
            "Major career changes and transitions",
        ],
        "variations": {
            "starting_from_wrist": (
                "A Fate line starting from the very base/wrist indicates "
                "early start in career, often from childhood. The person is "
                "self-made, starts working early, and builds career through "
                "personal effort from the very beginning."
            ),
            "starting_from_life_line": (
                "When Fate line rises from the Life line, the person's "
                "career is initially supported by family. Success comes "
                "through family help, inheritance, or family business."
            ),
            "starting_from_head_line": (
                "Fate line beginning at the Head line indicates late "
                "career success — around age 35-40. Success comes through "
                "intellectual pursuits and mental effort."
            ),
            "starting_from_heart_line": (
                "Fate line beginning at the Heart line indicates very late "
                "success — after age 45-50. May come through love, marriage, "
                "or emotional pursuit."
            ),
            "starting_from_moon": (
                "Fate line rising from the Moon mount indicates success "
                "through public favor, travel, or the influence of others, "
                "especially people from foreign lands."
            ),
            "deep_strong": (
                "A deep, strong Fate line indicates a powerful destiny "
                "controlled by external circumstances. The person's life "
                "follows a clear path set by fate. Great worldly success."
            ),
            "faint": (
                "A faint Fate line indicates a self-directed life where "
                "the person must create their own destiny through effort "
                "rather than relying on Lucky circumstances."
            ),
            "absent": (
                "No visible Fate line does NOT mean no career. It indicates "
                "a life of variety without a single fixed career path. "
                "The person creates their own destiny moment by moment."
            ),
            "broken": (
                "Breaks in the Fate line indicate career changes, job losses, "
                "or professional upheavals at those corresponding ages."
            ),
            "double": (
                "A double Fate line indicates two simultaneous career paths "
                "or a secondary income source that runs alongside the main career."
            ),
            "island": (
                "An island on the Fate line indicates a period of financial "
                "difficulty, career stagnation, or loss at that time."
            ),
            "branches_upward": (
                "Upward branches from the Fate line indicate promotions, "
                "raises, and career advancement at those periods."
            ),
        },
        "timing_method": (
            "Timing on the Fate line: Read from the base upward. The Head "
            "line crossing point marks approximately age 35. The Heart line "
            "crossing point marks approximately age 49-50. The tip of the "
            "line near Saturn finger marks age 70+."
        ),
    },
    "Sun_Apollo": {
        "hindi": "सूर्य रेखा",
        "location": "Runs toward the Sun/Apollo (ring) finger, parallel to Fate line",
        "governs": [
            "Fame, recognition, and public reputation",
            "Artistic talent and creative success",
            "Brilliance and personal magnetism",
            "Wealth through talent (as opposed to Fate line wealth through career)",
            "Celebrity status and public acclaim",
            "Quality of success — not just achievement but distinction",
        ],
        "variations": {
            "starting_from_wrist": (
                "Sun line from wrist to Apollo finger is extraordinarily rare "
                "and indicates a life of exceptional talent, fame, and "
                "recognition from very early age. A prodigy sign."
            ),
            "starting_from_head_line": (
                "Success through intellectual/creative effort coming in "
                "the mid-30s. Recognition earned through merit and talent."
            ),
            "starting_from_heart_line": (
                "Late recognition of talent, usually after age 50. "
                "The person may struggle for most of life before receiving "
                "appreciation. Talent recognized in mature years."
            ),
            "starting_from_fate_line": (
                "Recognition and fame tied directly to career success. "
                "The person becomes famous for their professional work."
            ),
            "absent": (
                "No Sun line means the person may be talented but will not "
                "receive public recognition or fame. A private life of "
                "achievement without public acclaim."
            ),
            "multiple_lines": (
                "Multiple Sun lines indicate scattered creative talents. "
                "The person is versatile but may not achieve greatness "
                "in any single field."
            ),
            "deep_strong": (
                "A deep, clear Sun line indicates extraordinary creative "
                "power and guaranteed public recognition."
            ),
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# MINOR LINES
# ═══════════════════════════════════════════════════════════════════════════

MINOR_LINES: Dict[str, Dict[str, Any]] = {
    "Marriage_Union": {
        "hindi": "विवाह रेखा",
        "location": "Horizontal lines on the percussion edge between Heart line and Mercury finger",
        "description": (
            "Marriage lines indicate significant emotional partnerships. "
            "The number, depth, and position of these lines reveal the "
            "nature and timing of major relationships."
        ),
        "variations": {
            "one_strong": "One deep, clear line indicates one significant marriage/union. Stable, lasting relationship.",
            "multiple": "Multiple lines indicate several significant relationships. Not all may be marriages.",
            "forked": "A forked Marriage line indicates separation or divorce.",
            "drooping_toward_heart": "Line drooping toward Heart line indicates the partner may pass away first.",
            "curving_upward": "Upward curve indicates a prosperous marriage. Partner brings happiness.",
            "island": "An island on the marriage line indicates a period of separation or difficulty in the marriage.",
            "absent": "No marriage lines does not mean no marriage — it may mean the emotional attachment is not deeply marked.",
        },
        "timing": (
            "The space between the Heart line and Mercury finger base represents "
            "approximately ages 15 to 70. A line halfway indicates marriage "
            "around age 25-30. Higher lines indicate later marriages."
        ),
    },
    "Children": {
        "hindi": "सन्तान रेखाएं",
        "location": "Fine vertical lines rising from the marriage line",
        "description": (
            "Fine vertical lines on the marriage line indicate children. "
            "Stronger, deeper lines indicate sons; finer, lighter lines "
            "indicate daughters. The number of lines indicates potential, "
            "not guaranteed children."
        ),
    },
    "Health_Hepatica": {
        "hindi": "स्वास्थ्य रेखा",
        "location": "Runs from the Mercury mount toward the base of the palm",
        "description": (
            "The Health line (also called Hepatica or Liver line) indicates "
            "the state of the digestive and nervous systems. Ironically, "
            "its ABSENCE is the best sign — it means perfect health."
        ),
        "variations": {
            "absent": "Best sign — indicates excellent health and robust constitution.",
            "wavy": "Digestive issues, liver/bile problems, chronic stomach complaints.",
            "broken": "Intermittent health issues, recurring illness.",
            "deep_red": "Fever tendency, inflammatory conditions.",
            "touching_life_line": "Warning sign: health issues serious enough to threaten vitality at that age.",
        },
    },
    "Girdle_of_Venus": {
        "hindi": "शुक्र वलय",
        "location": "Semicircle from between Jupiter-Saturn to between Apollo-Mercury",
        "description": (
            "The Girdle of Venus indicates heightened emotional and nervous "
            "sensitivity. In a good hand it enhances artistic perception. "
            "In a weak hand it indicates hysteria, nervous excess, and "
            "emotional instability."
        ),
    },
    "Ring_of_Solomon": {
        "hindi": "ज्ञान वलय",
        "location": "Semicircle at the base of the Jupiter finger",
        "description": (
            "The Ring of Solomon (or Ring of Jupiter) indicates exceptional "
            "wisdom, insight into human nature, and natural teaching or "
            "counseling ability. Found on the hands of sages and wise leaders."
        ),
    },
    "Bracelets_Rascettes": {
        "hindi": "मणिबंध रेखाएं",
        "location": "Horizontal lines on the wrist at the base of the palm",
        "description": (
            "Traditional palmistry reads each clear bracelet as approximately "
            "25-30 years of life. Three well-marked bracelets indicate "
            "75-90 years. The first bracelet also relates to health — "
            "if it arches upward into the palm, it may indicate health "
            "challenges, especially for women (reproductive health)."
        ),
    },
    "Travel": {
        "hindi": "यात्रा रेखाएं",
        "location": "Horizontal lines from the percussion edge into the Moon mount",
        "description": (
            "Travel lines indicate significant journeys, especially overseas. "
            "Deeper lines = more impactful journeys. Lines crossing the palm "
            "wide indicate journeys that change the person's life direction."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SPECIAL MARKS & SIGNS (Ch. 16–20)
# ═══════════════════════════════════════════════════════════════════════════

SPECIAL_MARKS: Dict[str, Dict[str, str]] = {
    "Star": {
        "symbol": "✦",
        "general": (
            "A star is a powerful sign that amplifies the energy of wherever "
            "it appears. On mounts, it intensifies the mount's qualities. "
            "On lines, it typically marks a sudden, dramatic event — "
            "brilliant success or shocking setback depending on position."
        ),
        "on_life_line": "Sudden shock or dramatic life event at that age.",
        "on_head_line": "Brilliant mental achievement or serious head injury.",
        "on_heart_line": "Spectacular love event or sudden heart-related incident.",
        "on_fate_line": "Dramatic career change — either a great rise or sudden fall.",
    },
    "Cross": {
        "symbol": "✚",
        "general": (
            "Crosses generally indicate obstacles, opposition, and difficulties. "
            "The exception is the cross on Jupiter mount (happy marriage) and "
            "the Mystic Cross between Head and Heart lines (psychic ability)."
        ),
        "mystic_cross": (
            "A cross between the Head and Heart lines (in the center of the "
            "Quadrangle) is called the Mystic Cross. It indicates strong "
            "psychic ability, interest in occult sciences, and intuitive "
            "powers. Cheiro considered this a mark of prophetic vision."
        ),
        "on_life_line": "Obstacles and difficulties at that period.",
        "on_head_line": "Mental crisis or head injury.",
        "on_heart_line": "Emotional crisis or broken relationship.",
    },
    "Square": {
        "symbol": "◻",
        "general": (
            "The square is the sign of protection and preservation. Wherever "
            "it appears, it guards against the negative effects of that area. "
            "A square on a break in a line repairs the damage of the break."
        ),
        "on_life_line": "Protection from illness or accident at that period.",
        "on_break": "Repairs broken line — prevents full impact of the disruption.",
    },
    "Triangle": {
        "symbol": "△",
        "general": (
            "Triangles on mounts indicate extraordinary talent and success "
            "in the field governed by that mount. They represent scientific "
            "or practical mastery of the mount's energy."
        ),
    },
    "Island": {
        "symbol": "◎",
        "general": (
            "Islands are always unfavorable. They indicate weakness, "
            "deception, or hereditary issues wherever they appear. On lines, "
            "they mark periods of difficulty proportional to their size."
        ),
        "on_life_line": "Extended illness or weakness lasting the duration of the island.",
        "on_head_line": "Mental confusion, headaches, or intellectual difficulty.",
        "on_heart_line": "Weakness of heart or emotional dishonesty.",
        "on_fate_line": "Financial loss or career difficulty.",
    },
    "Dot": {
        "symbol": "●",
        "general": (
            "Dots indicate sudden events — illness, shock, or injury. "
            "A red dot suggests inflammatory condition. A white dot "
            "suggests nervous condition."
        ),
        "on_life_line": "Sudden illness or accident at that age.",
        "on_head_line": "Head injury or sudden mental crisis.",
        "on_heart_line": "Emotional shock or heart condition.",
    },
    "Grid_Grille": {
        "symbol": "▦",
        "general": (
            "A grid or grille is formed by many lines crossing over a mount. "
            "It indicates negative expression of the mount's energy — "
            "excess, obstruction, or misdirection of the mount's qualities."
        ),
    },
    "Circle": {
        "symbol": "○",
        "general": (
            "Circles are rare. On the Sun mount, a circle indicates "
            "extraordinary fame and fortune. On most other locations, "
            "it indicates a period of confinement or limitation."
        ),
    },
    "Trident": {
        "symbol": "Ψ",
        "general": (
            "A trident at the end of any line is the most fortunate sign. "
            "On the Fate line ending at Saturn, it indicates fame, fortune, "
            "and achievement. On the Head line, it indicates brilliance. "
            "On the Heart line, it indicates fulfilling love."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# GREAT TRIANGLE & QUADRANGLE (Ch. 22)
# ═══════════════════════════════════════════════════════════════════════════

GREAT_TRIANGLE: Dict[str, str] = {
    "description": (
        "The Great Triangle is formed by the Life line, Head line, and Health "
        "line (Hepatica). Its shape and angles reveal fundamental character traits."
    ),
    "upper_angle": (
        "Formed where Life and Head lines meet. A clear, acute angle indicates "
        "fine intellect and delicate perception. A blunt angle indicates a "
        "slower, more practical mind."
    ),
    "middle_angle": (
        "Formed at the junction of Head and Health lines. A clear acute angle "
        "indicates long life and good health prospects."
    ),
    "lower_angle": (
        "Formed at the junction of Life and Health lines. Related to "
        "overall vitality and the body's recuperative powers."
    ),
    "large_triangle": "Indicates generous, broad-minded nature. Liberal thinking.",
    "small_triangle": "Indicates narrow-minded, mean-spirited nature.",
}

QUADRANGLE: Dict[str, str] = {
    "description": (
        "The Quadrangle is the space between the Head and Heart lines. "
        "Its shape indicates the balance between intellect and emotion."
    ),
    "wide_even": (
        "A wide, evenly-spaced Quadrangle indicates a broad-minded, just, "
        "and fair person. They balance logic and emotion well."
    ),
    "narrow": (
        "A narrow Quadrangle indicates a narrow-minded, prejudiced person. "
        "They are secretive, suspicious, and rigid in thinking."
    ),
    "wide_under_jupiter": (
        "Wider under Jupiter indicates independence in religious or "
        "philosophical thinking. Free-thinker in matters of belief."
    ),
    "wide_under_saturn": (
        "Wider under Saturn indicates independence from fatalistic thinking. "
        "The person believes in self-determination."
    ),
    "mystic_cross_present": (
        "A Mystic Cross within the Quadrangle is the mark of psychic "
        "ability and occult interest."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# TIMING SYSTEM — CHEIRO'S METHOD (Ch. 24)
# ═══════════════════════════════════════════════════════════════════════════

TIMING_SYSTEM: Dict[str, Any] = {
    "description": (
        "Cheiro developed a precise timing system for reading events on the "
        "palm. Each major line can be divided into time segments based on "
        "anatomical reference points. This system allows prediction of WHEN "
        "major events are likely to occur."
    ),
    "Life_line": {
        "method": (
            "Drop a vertical line from the center of the Saturn (middle) finger "
            "base to the Life line. Where it meets the Life line = age 20. "
            "Drop from the center between Saturn and Jupiter = age 10. "
            "The midpoint of the entire Life line = age 40. "
            "The line continues proportionally to the wrist (age 70-80+)."
        ),
        "landmarks": {
            "start": "Birth (age 0)",
            "head_line_separation": "Age of independence from family (typically 14-21)",
            "saturn_drop": "Age 20",
            "midpoint": "Age 40",
            "wrist_approach": "Age 70-80+",
        },
    },
    "Fate_line": {
        "method": (
            "The Fate line is read from bottom (wrist) to top (Saturn finger). "
            "Where the Head line crosses = age 35. "
            "Where the Heart line crosses = age 49-50. "
            "The tip near Saturn finger = age 70-75."
        ),
        "landmarks": {
            "wrist": "Age 0-5 (childhood)",
            "head_line_cross": "Age 35",
            "heart_line_cross": "Age 49-50",
            "saturn_base": "Age 70-75",
        },
    },
    "Head_line": {
        "method": (
            "Read from the start (under Jupiter) toward the percussion. "
            "A vertical line dropped from Saturn finger base = age 20-25. "
            "From Apollo finger base = age 35-40. "
            "From Mercury finger base = age 55-60."
        ),
    },
    "Heart_line": {
        "method": (
            "Read from under Mercury finger toward Jupiter. "
            "Under Mercury = ages 15-20. Under Apollo = ages 25-35. "
            "Under Saturn = ages 40-50. Under Jupiter = ages 55-70."
        ),
    },
    "prediction_framework": {
        "breaks": "Indicate sudden change at that time period",
        "islands": "Indicate difficulty lasting through the island's length",
        "branches_up": "Success and elevation at that period",
        "branches_down": "Loss and decline at that period",
        "color_change": "Health changes — deeper red = vigor, pale/blue = weakness",
        "stars": "Sudden dramatic event at that age",
        "crosses": "Obstacles or interference at that age",
        "squares": "Protection from danger at that age",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH DIAGNOSIS THROUGH PALM (Various chapters)
# ═══════════════════════════════════════════════════════════════════════════

HEALTH_INDICATORS: Dict[str, Dict[str, str]] = {
    "heart_disease": {
        "signs": (
            "Short, broad nails with bluish tinge. Heart line chained, broken, "
            "or islanded under Saturn or Apollo. Red/purple spots on Heart line. "
            "Blue tinge to nails."
        ),
        "traditional_advice": "Cheiro recommends moderation in diet and regular exercise.",
    },
    "nervous_disorders": {
        "signs": (
            "Chained Head line, numerous fine lines crossing the palm, "
            "fan-shaped nails, spotted nails (white dots), prominent "
            "Girdle of Venus. Over-developed Moon mount."
        ),
        "traditional_advice": "Rest, solitude, and avoidance of stimulants.",
    },
    "digestive_issues": {
        "signs": (
            "Strong Health line (Hepatica) — especially if wavy or reddish. "
            "Yellow tinge to palm and nails. Cross-bars on Health line."
        ),
        "traditional_advice": "Diet modification, avoiding rich foods and alcohol.",
    },
    "respiratory": {
        "signs": (
            "Long, narrow nails. Chained Life line in upper portion. "
            "Island at start of Life line. Narrow, flat Venus mount."
        ),
        "traditional_advice": "Fresh air, breathing exercises, avoiding damp environments.",
    },
    "brain_mental": {
        "signs": (
            "Islands, breaks, or stars on Head line. Head line fading or "
            "breaking at a point. Dot on Head line. Chained Head line "
            "under Saturn."
        ),
        "traditional_advice": "Mental rest, avoiding overwork, and intellectual moderation.",
    },
    "eye_problems": {
        "signs": (
            "Island or circle on Heart line under Apollo. Curved Head line "
            "with islands. Short nails with horizontal ridges."
        ),
        "traditional_advice": "Eye care and regular vision checks.",
    },
    "reproductive": {
        "signs": (
            "First bracelet (Rascette) arching upward into the palm. "
            "Islands on Life line in lower section. Weak Venus mount."
        ),
        "traditional_advice": "Gentle exercise and medical consultation.",
    },
    "longevity_indicators": {
        "positive": (
            "Long Life line reaching wrist, three clear bracelets, "
            "strong Fate line, clear Head line without breaks, "
            "pink nails, and firm Venus mount = long, healthy life."
        ),
        "caution": (
            "Short Life line with breaks, absent Fate line, chained "
            "Head line, bluish nails = lower vitality, needs care."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PERSONALITY PROFILING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

PERSONALITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "leader": {
        "indicators": [
            "Strong Jupiter mount", "Large thumb", "Long index finger",
            "Deep Head line", "Strong Fate line", "Square hand type",
        ],
        "description": (
            "Natural born leader with ambition, determination, and the "
            "ability to inspire and command others. Drawn to positions of "
            "authority and responsibility."
        ),
    },
    "creative_artist": {
        "indicators": [
            "Strong Sun/Apollo mount", "Conic/Psychic hand type",
            "Head line sloping to Moon", "Strong Venus mount",
            "Long ring finger", "Smooth finger joints",
        ],
        "description": (
            "Gifted with artistic vision and creative talent. Life revolves "
            "around beauty, self-expression, and the pursuit of aesthetic "
            "perfection."
        ),
    },
    "intellectual_scholar": {
        "indicators": [
            "Philosophic hand type", "Knotty finger joints",
            "Long Head line", "Strong Saturn mount",
            "Long middle finger", "Deep lines overall",
        ],
        "description": (
            "Deep thinker and lifelong learner. Driven by the quest for "
            "knowledge, understanding, and truth. Academic or research "
            "oriented."
        ),
    },
    "business_entrepreneur": {
        "indicators": [
            "Strong Mercury mount", "Long little finger",
            "Spatulate hand type", "Strong Fate and Sun lines",
            "Practical Head line", "Large thumb",
        ],
        "description": (
            "Shrewd business mind with commercial instinct and gift of "
            "persuasion. Creates wealth through trade, communication, "
            "and strategic thinking."
        ),
    },
    "healer_counselor": {
        "indicators": [
            "Medical stigmata (short vertical lines on Mercury mount)",
            "Strong Heart line", "Strong intuitive line (from Moon to Mercury)",
            "Philanthropic triangle on Jupiter",
            "Well-developed Moon mount",
        ],
        "description": (
            "Natural healer with deep empathy and intuitive understanding "
            "of others' suffering. Drawn to medicine, counseling, or "
            "spiritual healing."
        ),
    },
    "adventurer_explorer": {
        "indicators": [
            "Strong Moon mount", "Travel lines from percussion",
            "Spatulate fingertips", "Separated Life/Head lines",
            "Strong lower Mars", "Long Life line with fork at end",
        ],
        "description": (
            "Restless spirit driven by wanderlust and the need for new "
            "experiences. Life is defined by travel, exploration, and "
            "the pursuit of freedom."
        ),
    },
    "mystic_psychic": {
        "indicators": [
            "Psychic hand type", "Mystic Cross in Quadrangle",
            "Ring of Solomon", "Strong Moon mount",
            "Pointed fingertips", "Intuition line present",
        ],
        "description": (
            "Highly intuitive, spiritually evolved person with genuine "
            "psychic abilities. Drawn to occult sciences, mysticism, "
            "and the exploration of consciousness."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# TRAVEL & ACCIDENT ANALYSIS (Ch. 23)
# ═══════════════════════════════════════════════════════════════════════════

TRAVEL_ANALYSIS: Dict[str, str] = {
    "many_travel_lines": (
        "Numerous strong travel lines from the percussion into the Moon mount "
        "indicate a life filled with significant journeys. The person is "
        "a natural traveler whose life direction is shaped by voyages."
    ),
    "travel_affecting_fate": (
        "When a travel line extends and joins the Fate line, that particular "
        "journey will change the person's career or life direction permanently."
    ),
    "overseas_indicator": (
        "Life line forking toward the Moon mount at its end indicates "
        "emigration or permanent relocation to a foreign land."
    ),
    "accident_signs": (
        "A break in the Life line with a cross nearby, or a star on the "
        "Life line, indicates an accident at that age. Squares nearby protect "
        "from the worst outcome."
    ),
    "danger_from_water": (
        "A star or cross on the Moon mount indicates danger from water. "
        "This could mean drowning risk or health issues during sea travel."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPATIBILITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════

ELEMENT_COMPATIBILITY: Dict[str, Dict[str, str]] = {
    "Earth": {
        "Earth": "Stable and practical. May lack excitement but provides security.",
        "Air": "Air stimulates Earth's thinking, but may frustrate with lack of follow-through.",
        "Fire": "Challenging. Fire finds Earth too slow; Earth finds Fire too reckless.",
        "Water": "Nurturing combination. Water softens Earth, Earth grounds Water.",
    },
    "Air": {
        "Earth": "Earth provides stability Air needs, but Air may feel confined.",
        "Air": "Exciting intellectually but may lack grounding. All talk, little action.",
        "Fire": "Excellent combination. Air feeds Fire's passion with ideas and vision.",
        "Water": "Difficult. Air finds Water too emotional; Water finds Air too detached.",
    },
    "Fire": {
        "Earth": "Fire needs Earth's stability but chafes under its restrictions.",
        "Air": "Brilliant partnership. Air fans Fire's flames with fresh oxygen.",
        "Fire": "Passionate but volatile. Two fires can create an inferno or burn out.",
        "Water": "Water can extinguish Fire or create steam. Transformative but challenging.",
    },
    "Water": {
        "Earth": "Excellent. Earth contains Water, Water nourishes Earth.",
        "Air": "Difficult. Air creates waves in Water. Emotional turbulence.",
        "Fire": "Steam or extinguishment. Powerful transformation or destruction.",
        "Water": "Deep emotional bond. May drown in emotion together. Need external grounding.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PROFESSIONAL READING TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

def get_professional_greeting() -> str:
    """Returns a professional palm reader's greeting."""
    return (
        "Namaste 🙏 I am a palmistry guide trained in the complete methodology of "
        "Count Louis Hamon (Cheiro), the world's most celebrated palm reader who "
        "read the hands of kings, queens, prime ministers, and celebrities for over "
        "40 years. My analysis draws from Cheiro's comprehensive system covering "
        "hand types, finger analysis, line interpretation, mount reading, timing "
        "calculations, and mark interpretation.\n\n"
        "I will now provide your complete palm reading based on the detected lines "
        "and features. Remember — palmistry is an ancient interpretive tradition. "
        "The lines are tendencies, not certainties. Your free will always has the "
        "final word.\n\n"
        "What aspect of your reading would you like to explore?"
    )


def build_professional_system_prompt() -> str:
    """Returns the complete system prompt for the palm reading AI chatbot."""
    return """You are the world's most experienced palm reader, a digital incarnation of Cheiro (Count Louis Hamon) — the legendary palmist who read palms for over 40 years including Mark Twain, Oscar Wilde, Thomas Edison, the Prince of Wales, and numerous kings and prime ministers.

Your knowledge encompasses:
1. CHEIRO'S COMPLETE SYSTEM: 7 hand types (Elementary, Square, Spatulate, Philosophic, Conic, Psychic, Mixed), 7 mounts (Jupiter, Saturn, Sun/Apollo, Mercury, Mars Upper/Lower, Moon, Venus), all major lines (Life, Head, Heart, Fate, Sun), all minor lines (Marriage, Children, Health/Hepatica, Girdle of Venus, Ring of Solomon, Bracelets, Travel), and special marks (Star, Cross, Square, Triangle, Island, Dot, Grid, Circle, Trident).

2. TIMING PREDICTIONS: Use Cheiro's timing system to predict WHEN events occur:
   - Life line: Saturn finger drop = age 20, midpoint = age 40, wrist = 70-80
   - Fate line: Head line cross = age 35, Heart line cross = age 49-50
   - Head line: Saturn drop = age 20-25, Apollo drop = age 35-40
   - Heart line: Under Mercury = 15-20, Under Apollo = 25-35, Under Saturn = 40-50

3. READING STYLE: Give readings that are:
   - Detailed and specific (not vague)
   - Use exact timing when possible ("between ages 28-33 you may experience...")
   - Speak with authority and confidence
   - Use evocative, mystical language while remaining grounded
   - Always reference specific line features from the scan data
   - Provide both the traditional interpretation AND practical modern advice
   - Cover personality, career, love, health, finances, and spiritual growth
   - When asked about time, ALWAYS provide specific age ranges based on Cheiro's timing

4. ANSWER FORMAT: Structure responses with:
   - A brief mystical observation
   - The detailed Cheiro-based analysis referencing specific features
   - Time predictions when applicable
   - Practical modern wisdom
   - A thought-provoking closing insight

5. HEALTH DISCLAIMERS: When discussing health, always add: "This is traditional interpretive wisdom and should never replace professional medical advice."

6. NEVER say "I don't know" or "I'm not sure." A master palmist always has insight. If data is limited, provide general Cheiro-based wisdom for that question category.

You speak with the gravitas and wisdom of 40+ years of professional experience. Your readings should feel like consulting the world's greatest living palmist."""
