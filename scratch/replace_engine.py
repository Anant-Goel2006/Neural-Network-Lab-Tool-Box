import os

with open(r'utils/palmistry_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    report["chat_context"] = palm_report_to_chat_context(report)
    return report"""

replacement = """    report["chat_context"] = palm_report_to_chat_context(report)
    
    # Generate AI Reading Summary and Predictions
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
        print("AI generation failed:", e)

    return report"""

if target in content:
    content = content.replace(target, replacement)
    with open(r'utils/palmistry_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')
else:
    print('FAIL')
