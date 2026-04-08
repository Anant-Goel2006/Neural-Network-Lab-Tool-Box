import streamlit as st
import streamlit.components.v1 as components
import json

def render_voice_button(text_to_speak, key_suffix=""):
    """
    Renders an HTML button that uses the browser's native Web Speech API 
    to read the provided text aloud.
    """
    safe_text = json.dumps(text_to_speak)
    
    html_code = f"""
    <div style="font-family: 'Montserrat', sans-serif;">
        <button id="playBtn_{key_suffix}" style="
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
            outline: none;
        " onmouseover="this.style.transform='scale(1.05)';" onmouseout="this.style.transform='scale(1.0)';">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
            Play Audio
        </button>
    </div>
    
    <script>
        document.getElementById('playBtn_{key_suffix}').addEventListener('click', function() {{
            window.speechSynthesis.cancel(); // Stop any current speech
            let utterance = new SpeechSynthesisUtterance({safe_text});
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            // Optionally try to find a natural sounding English voice
            let voices = window.speechSynthesis.getVoices();
            let englishVoices = voices.filter(v => v.lang.startsWith('en'));
            if(englishVoices.length > 0) {{
                // Try to prefer a female/natural voice if available, else first english
                let preferred = englishVoices.find(v => v.name.includes('Google') || v.name.includes('Natural'));
                if(preferred) utterance.voice = preferred;
                else utterance.voice = englishVoices[0];
            }}
            window.speechSynthesis.speak(utterance);
        }});
        
        // Ensure voices are loaded (Chrome edge case)
        window.speechSynthesis.onvoiceschanged = function() {{
            window.speechSynthesis.getVoices();
        }};
    </script>
    """
    
    components.html(html_code, height=45)
