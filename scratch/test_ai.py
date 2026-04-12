import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils.ai_helper import get_selected_provider, get_api_key

def test_ai_detection():
    print("--- AI Auto-Detection Test ---")
    
    # Mock environment keys
    os.environ["NVIDIA_API_KEY"] = "mock_nv_key"
    os.environ["OPENAI_API_KEY"] = ""
    
    provider = get_selected_provider()
    key = get_api_key(provider)
    print(f"Test 1 (NVIDIA only): Detected Provider: {provider}, Key found: {bool(key)}")
    assert provider == "NVIDIA"
    
    # Mock OpenAI only
    os.environ["NVIDIA_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = "mock_oa_key"
    
    provider = get_selected_provider()
    key = get_api_key(provider)
    print(f"Test 2 (OpenAI only): Detected Provider: {provider}, Key found: {bool(key)}")
    assert provider == "OpenAI"
    
    # Mock both (Prioritize NVIDIA)
    os.environ["NVIDIA_API_KEY"] = "mock_nv_key"
    os.environ["OPENAI_API_KEY"] = "mock_oa_key"
    
    provider = get_selected_provider()
    print(f"Test 3 (Both): Detected Provider: {provider}")
    assert provider == "NVIDIA"

    print("Success!")

if __name__ == "__main__":
    test_ai_detection()
