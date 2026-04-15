import json

import streamlit as st


def render_voice_button(text_to_speak, key_suffix=""):
    safe_text = json.dumps(text_to_speak)

    html_code = f"""
    <div style="font-family: 'Montserrat', sans-serif; margin: 6px 0 10px 0;">
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
            <button id="playBtn_{key_suffix}" style="
                background: linear-gradient(135deg, #3B82F6, #8B5CF6);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 50px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35);
                transition: all 0.2s ease;
            ">Play Audio</button>
            <button id="stopBtn_{key_suffix}" style="
                background: rgba(15,23,42,0.9);
                color: #E2E8F0;
                border: 1px solid rgba(148,163,184,0.25);
                padding: 8px 14px;
                border-radius: 50px;
                font-size: 13px;
                cursor: pointer;
            ">Stop</button>
            <select id="voiceSelect_{key_suffix}" style="
                background: rgba(15,23,42,0.9);
                color: #E2E8F0;
                border: 1px solid rgba(148,163,184,0.25);
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 13px;
                min-width: 160px;
            "></select>
            <select id="rateSelect_{key_suffix}" style="
                background: rgba(15,23,42,0.9);
                color: #E2E8F0;
                border: 1px solid rgba(148,163,184,0.25);
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 13px;
            ">
                <option value="0.85">Slow</option>
                <option value="1.0" selected>Normal</option>
                <option value="1.15">Fast</option>
            </select>
            <select id="pitchSelect_{key_suffix}" style="
                background: rgba(15,23,42,0.9);
                color: #E2E8F0;
                border: 1px solid rgba(148,163,184,0.25);
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 13px;
            ">
                <option value="0.9">Low Pitch</option>
                <option value="1.0" selected>Normal Pitch</option>
                <option value="1.1">High Pitch</option>
            </select>
        </div>
    </div>

    <script>
        (function() {{
            const rawText = {safe_text};
            const playBtn = document.getElementById('playBtn_{key_suffix}');
            const stopBtn = document.getElementById('stopBtn_{key_suffix}');
            const voiceSelect = document.getElementById('voiceSelect_{key_suffix}');
            const rateSelect = document.getElementById('rateSelect_{key_suffix}');
            const pitchSelect = document.getElementById('pitchSelect_{key_suffix}');

            function normalizeSpeechText(text) {{
                return String(text)
                    .replace(/ŷ/g, ' y hat ')
                    .replace(/η/g, ' eta ')
                    .replace(/Δ/g, ' delta ')
                    .replace(/σ/g, ' sigma ')
                    .replace(/⊙/g, ' x nor ')
                    .replace(/⊕/g, ' or ')
                    .replace(/⊗/g, ' and ')
                    .replace(/⊻/g, ' x or ')
                    .replace(/·/g, ' times ')
                    .replace(/→/g, ' leads to ')
                    .replace(/≤/g, ' less than or equal to ')
                    .replace(/≥/g, ' greater than or equal to ')
                    .replace(/≈/g, ' approximately ')
                    .replace(/%/g, ' percent ')
                    .replace(/w1/g, ' w 1 ')
                    .replace(/w2/g, ' w 2 ')
                    .replace(/x1/g, ' x 1 ')
                    .replace(/x2/g, ' x 2 ')
                    .replace(/dL\\/dW/g, ' d L by d W ')
                    .replace(/dL\\/dA/g, ' d L by d A ')
                    .replace(/dA\\/dZ/g, ' d A by d Z ')
                    .replace(/dL\\/dZ/g, ' d L by d Z ')
                    .replace(/\\s+/g, ' ')
                    .trim();
            }}

            function preferredVoices(voices) {{
                const english = voices.filter(v => v.lang && v.lang.toLowerCase().startsWith('en'));
                return english.length ? english : voices;
            }}

            function populateVoices() {{
                const voices = preferredVoices(window.speechSynthesis.getVoices());
                const savedVoice = localStorage.getItem('neurolab_voice_name') || '';
                const currentValue = voiceSelect.value || savedVoice;
                voiceSelect.innerHTML = '';
                voices.forEach((voice, idx) => {{
                    const option = document.createElement('option');
                    option.value = voice.name;
                    option.textContent = `${{voice.name}} (${{voice.lang}})`;
                    if ((currentValue && voice.name === currentValue) || (!currentValue && idx === 0)) {{
                        option.selected = true;
                    }}
                    voiceSelect.appendChild(option);
                }});
            }}

            rateSelect.value = localStorage.getItem('neurolab_voice_rate') || '1.0';
            pitchSelect.value = localStorage.getItem('neurolab_voice_pitch') || '1.0';

            populateVoices();
            if (window.speechSynthesis.onvoiceschanged !== undefined) {{
                window.speechSynthesis.onvoiceschanged = populateVoices;
            }}

            voiceSelect.addEventListener('change', function() {{
                localStorage.setItem('neurolab_voice_name', voiceSelect.value);
            }});
            rateSelect.addEventListener('change', function() {{
                localStorage.setItem('neurolab_voice_rate', rateSelect.value);
            }});
            pitchSelect.addEventListener('change', function() {{
                localStorage.setItem('neurolab_voice_pitch', pitchSelect.value);
            }});

            playBtn.addEventListener('click', function() {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(normalizeSpeechText(rawText));
                utterance.rate = parseFloat(rateSelect.value || '1.0');
                utterance.pitch = parseFloat(pitchSelect.value || '1.0');
                const voices = preferredVoices(window.speechSynthesis.getVoices());
                const selected = voices.find(v => v.name === voiceSelect.value);
                if (selected) {{
                    utterance.voice = selected;
                }} else if (voices.length) {{
                    utterance.voice = voices[0];
                }}
                window.speechSynthesis.speak(utterance);
            }});

            stopBtn.addEventListener('click', function() {{
                window.speechSynthesis.cancel();
            }});
        }})();
    </script>
    """

    st.html(html_code)
