import hashlib
import json
import uuid

import streamlit as st

from utils.ai_helper import get_ai_explanation
from utils.voice import render_voice_button


def push_tutor_insight(text, context="AI Tutor Insight"):
    """
    Pushes an external insight into the chatbot's history if it doesn't already exist.
    Uses a small hash of the text to ensure uniqueness across reruns.
    """
    page_name = st.session_state.get("last_visited_page", "global")
    state_key = f"chat_history_{page_name}"

    if state_key not in st.session_state:
        st.session_state[state_key] = []

    seen_key = f"seen_insights_{page_name}"
    if seen_key not in st.session_state:
        st.session_state[seen_key] = set()

    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash not in st.session_state[seen_key]:
        st.session_state[state_key].append(
            {
                "role": "assistant",
                "content": text,
                "label": context,
                "id": str(uuid.uuid4())[:8],
            }
        )
        st.session_state[seen_key].add(text_hash)


def _stringify_context(context_payload):
    if context_payload is None:
        return ""
    if isinstance(context_payload, str):
        return context_payload
    try:
        return json.dumps(context_payload, indent=2)
    except TypeError:
        return str(context_payload)


def _default_local_reply(context_description, context_payload):
    context_text = _stringify_context(context_payload)
    if context_text:
        return (
            f"I can still help locally with {context_description}. "
            f"Here is the current context I have: {context_text[:700]}"
        )
    return (
        f"I can help explain {context_description}, but the live AI provider is not responding right now. "
        "Please ensure your API key (NVIDIA or OpenAI) is correctly configured in your environment settings (e.g., .env or Streamlit secrets)."
    )


def render_chatbot(
    context_description="the current neural network state",
    context_payload=None,
    system_prompt=None,
    fallback_builder=None,
    greeting=None,
    theme=None,
    tutor_label="NEURAL AI TUTOR",
    placeholder="Message your AI Tutor...",
):
    """
    Renders a premium AI Tutor Chatbot.
    theme: optional Module_Theme dict (primary_color, gradient, etc.)
    tutor_label: header label string
    placeholder: chat input placeholder text
    """
    st.divider()

    # Derive styling from theme
    border_color = theme["primary_color"] if theme else "#3B82F6"
    header_gradient = theme["gradient"] if theme else "linear-gradient(90deg, #3B82F6, #8B5CF6)"

    page_name = st.session_state.get("last_visited_page", "global")
    state_key = f"chat_history_{page_name}"
    default_greeting = greeting or (
        f"Hello! I am your {tutor_label.title()}. I am here to guide you through {context_description}. "
        "What concept shall we explore first?"
    )

    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {
                "role": "assistant",
                "content": default_greeting,
                "label": f"{tutor_label} — Welcome",
                "id": "init",
            }
        ]

    with st.container(border=True):
        # Sci-Fi Jarvis style CSS injection
        st.markdown("""
            <style>
                /* Avatar customizations */
                .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-1c7y2kd { /* Default bot icon container usually */
                    background-color: transparent !important;
                }
                .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-p4micv {
                    background-color: transparent !important;
                }
                
                /* Chat bubble styling */
                .stChatMessage {
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                    border: 1px solid rgba(0, 255, 255, 0.2);
                    background-color: rgba(10, 20, 30, 0.6) !important;
                    box-shadow: 0 0 10px rgba(0, 255, 255, 0.05), inset 0 0 15px rgba(0, 255, 255, 0.05);
                    backdrop-filter: blur(5px);
                    color: #E0FFFF;
                    font-family: 'Consolas', 'Courier New', monospace;
                }
                
                /* Assistant message specific */
                .stChatMessage:has([data-testid="stIconMaterial"][title="assistant"]),
                .stChatMessage:nth-child(odd) {
                    border-left: 3px solid #00FFFF;
                    border-right: 1px solid rgba(0, 255, 255, 0.1);
                    background: linear-gradient(90deg, rgba(0, 255, 255, 0.1) 0%, rgba(10, 20, 30, 0.8) 20%);
                }
                
                /* User message specific */
                .stChatMessage:has([data-testid="stIconMaterial"][title="user"]),
                .stChatMessage:nth-child(even) {
                    border-right: 3px solid #8B5CF6;
                    border-left: 1px solid rgba(139, 92, 246, 0.1);
                    background: linear-gradient(270deg, rgba(139, 92, 246, 0.1) 0%, rgba(10, 20, 30, 0.8) 20%);
                    color: #F3E8FF;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background: {header_gradient}; border-radius: 8px 8px 0 0;
                        padding: 12px 18px; margin: -1px -1px 12px -1px;
                        border-left: 4px solid {border_color};">
                <div style="font-family:'Montserrat',sans-serif; font-weight:800;
                            font-size:15px; color:#FFFFFF; letter-spacing:2px;
                            text-transform:uppercase; text-shadow: 0 0 8px rgba(255,255,255,0.6);">
                    🤖 {tutor_label} [SYSTEM ONLINE]
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption(f"Context: {context_description}")

        chat_container = st.container(height=400)
        with chat_container:
            history = st.session_state[state_key]
            
            # Find the index of the last assistant message
            last_assistant_idx = -1
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "assistant":
                    last_assistant_idx = i
                    break
                    
            for i, message in enumerate(history):
                with st.chat_message(message["role"]):
                    if "label" in message:
                        st.markdown(f"**{message['label']}**")
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and i == last_assistant_idx:
                        render_voice_button(
                            message["content"],
                            key_suffix=f"chat_{message.get('id', 'default')}",
                        )

        if prompt := st.chat_input(placeholder):
            st.session_state[state_key].append(
                {"role": "user", "content": prompt, "id": str(uuid.uuid4())[:8]}
            )

            payload_text = _stringify_context(context_payload)
            # Detect palm reading mode for longer, more professional responses
            is_palm = "palm" in context_description.lower() or "cheiro" in (system_prompt or "").lower()
            instruction = (
                "Give a detailed, professional palm reading response with specific insights, "
                "timing predictions, and practical advice. Use 6 to 15 rich sentences."
            ) if is_palm else (
                "Answer as a patient expert tutor in 3 to 6 clear sentences."
            )
            full_prompt = (
                f"Context Description: {context_description}\n"
                f"Current Module State:\n{payload_text or 'No structured state was supplied.'}\n\n"
                f"User Question: {prompt}\n"
                f"{instruction}"
            )

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_ai_explanation(
                            full_prompt,
                            system_prompt=system_prompt
                            or (
                                "You are a brilliant AI tutor in a neural network lab. "
                                "Use simple analogies for beginners and stay grounded in the supplied context."
                            ),
                            max_tokens=800 if is_palm else 400,
                        )

                        if not response and fallback_builder is not None:
                            try:
                                response = fallback_builder(prompt)
                            except Exception:
                                response = None

                        if not response:
                            response = _default_local_reply(context_description, context_payload)

                        st.markdown(response)
                        msg_id = str(uuid.uuid4())[:8]
                        st.session_state[state_key].append(
                            {
                                "role": "assistant",
                                "content": response,
                                "label": "Tutor Response",
                                "id": msg_id,
                            }
                        )
                        render_voice_button(response, key_suffix=f"chat_{msg_id}")

            st.rerun()
