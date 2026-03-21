import streamlit as st
import time
import plotly.graph_objects as go

def inject_global_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Luckiest+Guy&family=Anton&family=Inter:wght@400;700&family=Roboto+Mono&display=swap');

        /* ──── NEUROLAB: COSMIC GRAPHIC NOVEL ──── */
        .stApp {
            background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #020617 100%);
            background-attachment: fixed;
            color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }

        /* ──── COSMIC STARFIELD OVERLAY ──── */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: 
                radial-gradient(1px 1px at 20px 30px, #fff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 40px 70px, #fff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 50px 160px, #fff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 90px 40px, #fff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 130px 80px, #fff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 160px 120px, #fff, rgba(0,0,0,0));
            background-repeat: repeat;
            background-size: 200px 200px;
            opacity: 0.3;
            pointer-events: none;
            z-index: 0;
        }

        /* ──── HALFTONE PATTERN ──── */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: radial-gradient(#ffffff0a 1px, transparent 0);
            background-size: 4px 4px;
            pointer-events: none;
            z-index: 1;
        }

        /* ──── GRAPHIC NOVEL CARDS ──── */
        .premium-card {
            background: #0f172a;
            border: 4px solid #000000;
            border-radius: 0px; /* Sharp comic look */
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 8px 8px 0px #000000;
            transition: transform 0.1s ease;
            position: relative;
            z-index: 2;
        }
        
        .premium-card:hover {
            transform: translate(-2px, -2px);
            box-shadow: 10px 10px 0px #3B82F6;
        }

        h1, h2, h3 {
            font-family: 'Bangers', cursive !important;
            color: #FFFFFF !important;
            text-shadow: 3px 3px 0px #000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000;
            letter-spacing: 2px !important;
            text-transform: uppercase;
        }

        /* ──── COMIC ACTION BUTTONS ──── */
        .stButton > button {
            background: #EF4444 !important; /* Action Red */
            border: 3px solid #000 !important;
            border-radius: 0px !important;
            color: #fff !important;
            font-family: 'Luckiest Guy', cursive !important;
            font-size: 20px !important;
            text-transform: uppercase !important;
            box-shadow: 4px 4px 0px #000 !important;
            padding: 10px 24px !important;
            transition: all 0.1s ease !important;
        }
        .stButton > button:hover {
            background: #FACC15 !important; /* Superhero Yellow */
            color: #000 !important;
            transform: translate(-1px, -1px);
            box-shadow: 6px 6px 0px #000 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #020617 !important;
            border-right: 4px solid #000;
        }

        /* Customize Metric */
        [data-testid="stMetricValue"] {
            font-family: 'Anton', sans-serif !important;
            font-size: 38px !important;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:40px; text-align: left; transform: skewX(-2deg);">
            <h2 style="margin:0; font-size:42px; border-bottom: 6px solid #FACC15; display:inline-block; padding-right:20px;">{title}</h2>
            <p style="color:#94A3B8; font-size:18px; font-weight:700; margin-top:8px; font-family:'Luckiest Guy'; text-transform:uppercase;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: #1e1b4b; border: 5px solid #000; padding:40px; 
            margin-bottom: 50px; box-shadow: 12px 12px 0px #EF4444; position:relative;">
            <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                <div style="font-size: 80px; filter: drop-shadow(4px 4px 0px #000);">{icon}</div>
                <div style="flex:1;">
                    <h1 style="font-size: 64px; margin: 0; line-height:1;">{title}</h1>
                    <p style="color:#FACC15; font-size:20px; font-family:'Luckiest Guy', cursive; letter-spacing:1px; margin-top:10px;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#EF4444", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 24, 'color': '#FFFFFF', 'family': 'Bangers'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 3, 'tickcolor': "#000"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.5)",
            'borderwidth': 4,
            'bordercolor': "#000",
            'steps': [{'range': [0, max_val], 'color': 'rgba(250,204,21,0.2)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bangers"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #F8FAFC; font-family: \'Roboto Mono\', monospace; font-size: 14px; margin-bottom: 6px; border-bottom: 1px dashed #334155; padding-bottom:4px;">'
                        f'<span style="color: #FACC15; font-weight:bold;">[{time.strftime("%H:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: #020617; border: 4px solid #000; padding: 25px; box-shadow: 6px 6px 0px #000;">
            <div style="font-family: 'Bangers', cursive; font-size: 24px; color: #EF4444; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px;">// Mission Log</div>
            {log_html if logs else '<div style="color: #475569; font-family: Roboto Mono, monospace;">Standby for signal...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#3B82F6"):
    st.markdown(f"""
        <div style="background: #0f172a; border: 4px solid #000; border-right: 12px solid {clr}; padding: 30px; margin-bottom: 40px; box-shadow: 10px 10px 0px rgba(0,0,0,0.5);">
            <div style="font-family:'Bangers', cursive; font-size:28px; color:{clr}; letter-spacing: 2px; margin-bottom:12px;">
                {label}
            </div>
            <div style="font-size:18px; color:#F8FAFC; font-family:'Inter', sans-serif; line-height: 1.5; font-weight: 500;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
