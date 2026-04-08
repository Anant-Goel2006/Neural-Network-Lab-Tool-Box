import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

def get_api_key():
    k = os.getenv("NVIDIA_API_KEY")
    if k and k.strip():
        return k.strip()
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass
    return None

def get_ai_explanation(prompt, system_prompt="You are an expert AI tutor explaining deep learning concepts visually and clearly. Keep explanations concise, around 2-3 sentences max.", model="meta/llama-3.1-8b-instruct", max_tokens=150):
    """
    Sends a prompt to the NVIDIA LLM API and returns the textual explanation.
    Returns None if the API fails or is unavailable.
    """
    key = get_api_key()
    if not key or not OPENAI_OK:
        return None

    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI Helper Error: {e}")
        return None
