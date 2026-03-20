import streamlit as st
import time
import plotly.graph_objects as go

def inject_global_css():
    """Injects the Premium Justice Suite (Light/Bold) theme CSS."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@400;700;900&display=swap');
        
        .stApp {
            background-color: #FFFFFF;
            color: #121212;
            font-family: 'Inter', sans-serif;
        }

        /* ──── PREMIUM CARS ──── */
        .premium-card {
            background: #FFFFFF;
            border: 3px solid #121212;
            box-shadow: 6px 6px 0px #121212;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.2s;
        }
        .premium-card:hover {
            transform: translate(-2px, -2px);
            box-shadow: 10px 10px 0px #121212;
        }

        h1, h2, h3 {
            font-family: 'Bangers', cursive !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #121212;
        }

        /* ──── BUTTONS ──── */
        .stButton > button {
            border: 3px solid #121212 !important;
            border-radius: 0px !important;
            text-transform: uppercase !important;
            font-weight: 900 !important;
            letter-spacing: 1px !important;
            box-shadow: 4px 4px 0px #121212 !important;
            transition: all 0.1s !important;
        }
        .stButton > button:active {
            box-shadow: 0px 0px 0px #121212 !important;
            transform: translate(4px, 4px) !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 2px solid #121212;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, subtitle):
    """Justice Suite section header."""
    st.markdown(f"""
        <div style="margin-bottom: 25px; border-left: 8px solid #005BEA; padding-left: 15px;">
            <h2 style="margin: 0; color: #121212; font-size: 28px;">{title.upper()}</h2>
            <p style="color: #64748B; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 2px;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    """Justice Suite Main Header."""
    st.markdown(f"""
        <div style="background: #FFFFFF; border: 4px solid #121212; padding: 30px; 
            box-shadow: 8px 8px 0px #ED1D24; margin-bottom: 40px; text-align: center;">
            <div style="font-size: 50px; margin-bottom: 10px;">{icon}</div>
            <h1 style="font-size: 52px; margin: 0; color: #121212;">{title.upper()}</h1>
            <p style="color: #005BEA; font-size: 14px; letter-spacing: 4px; font-weight: 900; margin-top: 10px; text-transform: uppercase;">
                {sub}
            </p>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#005BEA", height=220):
    """Justice Suite Speedometer."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#121212', 'family': 'Impact'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 2, 'tickcolor': "#121212"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#121212",
            'steps': [{'range': [0, max_val], 'color': '#F1F5F9'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#121212", 'family': "Impact"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    """Justice Suite Log."""
    log_html = "".join([f'<div style="color: #121212; font-family: monospace; font-size: 12px; margin-bottom: 4px; border-bottom: 1px dashed #DDD;">'
                        f'<span style="color: #005BEA; font-weight: 900;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-10:]])
    placeholder.markdown(f"""
        <div style="background: #FFF; border: 3px solid #121212; padding: 15px; box-shadow: 5px 5px 0px #64748B;">
            <div style="font-family: 'Impact'; font-size: 14px; color: #121212; margin-bottom: 10px; text-transform: uppercase;">Engine Terminal</div>
            {log_html if logs else '<div style="color: #94A3B8;">Initializing...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#005BEA"):
    """Justice Suite NLP Insight."""
    st.markdown(f"""
        <div style="background:#FFF; border:3px solid #121212; padding:15px; box-shadow:5px 5px 0px {clr};">
            <div style="font-family:'Impact'; font-size:12px; color:{clr}; text-transform:uppercase;">{label}</div>
            <div style="font-size:16px; font-weight:800; color:#121212;">{text}</div>
        </div>
    """, unsafe_allow_html=True)
