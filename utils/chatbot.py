import streamlit as st
import os
import uuid
import hashlib
from utils.ai_helper import get_ai_explanation
from utils.voice import render_voice_button

def push_tutor_insight(text, context="AI Tutor Insight"):
    """
    Pushes an external insight into the chatbot's history if it doesn't already exist.
    Uses a small hash of the text to ensure uniqueness across reruns.
    """
    page_name = st.session_state.get('last_visited_page', 'global')
    state_key = f"chat_history_{page_name}"
    
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    
    # Check if this specific insight was already pushed in this session
    # We use a set of seen insight hashes in session_state
    seen_key = f"seen_insights_{page_name}"
    if seen_key not in st.session_state:
        st.session_state[seen_key] = set()
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash not in st.session_state[seen_key]:
        st.session_state[state_key].append({
            "role": "assistant", 
            "content": text, 
            "label": context,
            "id": str(uuid.uuid4())[:8]
        })
        st.session_state[seen_key].add(text_hash)

def render_chatbot(context_description="the current neural network state"):
    """
    Renders a premium AI Tutor Chatbot.
    """
    st.divider()
    
    # Session state for chat history
    state_key = f"chat_history_{st.session_state.get('last_visited_page', 'global')}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {"role": "assistant", "content": f"Greeting! I am your Neural Tutor. I'm here to guide you through {context_description}. What concept shall we explore first?", "label": "AI Tutor Startup", "id": "init"}
        ]

    # Use a more premium layout for the chatbot
    with st.container(border=True):
        st.markdown('<div class="tutor-header">🤖 NEURAL AI TUTOR</div>', unsafe_allow_html=True)
        st.caption(f"Context: {context_description.title()}")
        
        chat_container = st.container(height=400)
        
        # Display chat history with premium styling and audio
        with chat_container:
            for message in st.session_state[state_key]:
                with st.chat_message(message["role"]):
                    if "label" in message:
                        st.markdown(f"**{message['label']}**")
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        render_voice_button(message["content"], key_suffix=f"chat_{message.get('id', 'default')}")

        # Chat input at the bottom of the container
        if prompt := st.chat_input("Message your AI Tutor..."):
            st.session_state[state_key].append({"role": "user", "content": prompt, "id": str(uuid.uuid4())[:8]})
            
            # Send to AI and get response
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        full_prompt = f"Context: The user is currently exploring {context_description}. Participant's Question: {prompt}. Answer as a professional AI tutor in 3-4 sentences."
                        response = get_ai_explanation(full_prompt, system_prompt="You are a brilliant AI tutor in a neural network lab. Use simple analogies for beginners.")
                        
                        if not response:
                            response = "I encountered a minor neural glitch. Please check your AI API key or connection."
                        
                        st.markdown(response)
                        msg_id = str(uuid.uuid4())[:8]
                        st.session_state[state_key].append({
                            "role": "assistant", 
                            "content": response, 
                            "label": "Tutor Response",
                            "id": msg_id
                        })
                        render_voice_button(response, key_suffix=f"chat_{msg_id}")
            st.rerun()
