import os

with open(r'utils/palmistry_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    # Generate AI Reading Summary and Predictions
    try:
        from utils.ai_helper import get_ai_explanation
        ai_prompt = (
            "You are an expert palm reader. Based on the following extracted palmistry metrics, "
            "provide a comprehensive reading. Structure your response EXACTLY as follows:\\n\\n"
            "🔮 **AI READING SUMMARY**\\n(Write a 3-sentence high-level summary of the person's character and life path based on the dominant mounts and line ratios.)\\n\\n"
            "🔍 **DEEP DIVE ANALYSIS**\\n(Analyze the Life, Head, and Heart lines in detail, highlighting unique features like islands, branches, and depth.)\\n\\n"
            "⏳ **PREDICTIONS**\\n(Give 2 specific life predictions or actionable advice based on the data.)\\n\\n"
            "Data:\\n" + str(report["chat_context"])[:1500]
        )
        ai_response = get_ai_explanation(
            ai_prompt, 
            system_prompt="You are a professional palm reader.", 
            max_tokens=600
        )
        if ai_response:
            report["summary"] = ai_response
    except Exception as e:
        print("AI generation failed:", e)"""

replacement = """    # Generate Full AI Reading
    try:
        from utils.ai_helper import get_ai_explanation
        ai_prompt = (
            "You are an expert palm reader. Based on the following extracted palmistry metrics, "
            "provide a comprehensive reading using beautiful Markdown formatting. Structure your response EXACTLY as follows:\\n\\n"
            "## 🔮 Professional Reading Summary\\n(Write a 3-4 sentence high-level summary of the person's character and life path based on the dominant mounts and line ratios.)\\n\\n"
            "## 🔍 Deep Dive Analysis\\n(Analyze the Life, Head, and Heart lines in detail. Highlight unique features like islands, branches, breaks, and depth.)\\n\\n"
            "## ⛰️ Mount Analysis\\n(Discuss their dominant mounts and what it means for their personality and career.)\\n\\n"
            "## ⏳ Time Predictions\\n(Give 2-3 specific life predictions, age ranges, or actionable advice based on the data.)\\n\\n"
            "Data:\\n" + str(report["chat_context"])[:2000]
        )
        ai_response = get_ai_explanation(
            ai_prompt, 
            system_prompt="You are an elite, highly insightful professional palm reader. Your readings are structured, profound, and visually engaging using Markdown emojis and bold text.", 
            max_tokens=1500
        )
        if ai_response:
            report["full_ai_reading"] = ai_response
    except Exception as e:
        print("AI generation failed:", e)"""

if target in content:
    content = content.replace(target, replacement)
    with open(r'utils/palmistry_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')
else:
    print('FAIL')
