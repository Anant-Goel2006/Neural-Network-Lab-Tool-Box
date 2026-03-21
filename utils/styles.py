import streamlit as st
import time
import plotly.graph_objects as go

def inject_global_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;500;700&family=Inter:wght@400;600&display=swap');
        
        .stApp {
            background-color: #050505;
            color: #E4E4E7;
            font-family: 'Inter', sans-serif;
        }

        /* ──── GRAPHIC NOVEL PANELS ──── */
        .premium-card {
            background: #0A0A0A;
            border: 2px solid #27272A;
            box-shadow: 5px 5px 0px #E11D48;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.2s;
        }
        .premium-card:hover {
            transform: translate(-2px, -2px);
            box-shadow: 7px 7px 0px #E11D48;
            border-color: #3F3F46;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #FAFAFA;
        }

        /* ──── BUTTONS ──── */
        .stButton > button {
            background-color: #0A0A0A !important;
            border: 2px solid #3F3F46 !important;
            border-radius: 0px !important;
            color: #FAFAFA !important;
            font-weight: 500 !important;
            font-size: 16px !important;
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.15s ease-in-out !important;
            padding: 8px 24px !important;
            box-shadow: 3px 3px 0px #27272A !important;
        }
        .stButton > button:hover {
            border-color: #E11D48 !important;
            box-shadow: 4px 4px 0px #E11D48 !important;
            transform: translate(-1px, -1px) !important;
        }
        .stButton > button:active {
            transform: translate(2px, 2px) !important;
            box-shadow: 1px 1px 0px #E11D48 !important;
        }

        /* ──── HEADER MENU BAR ──── */
        header[data-testid="stHeader"] {
            background-color: #050505 !important;
            border-bottom: 2px solid #E11D48 !important;
            box-shadow: none !important;
        }
        header[data-testid="stHeader"] * {
            color: #A1A1AA !important;
            font-family: 'Oswald', sans-serif !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: #0A0A0A !important;
            border-right: 2px solid #27272A;
        }
        
        hr {
            border-top: 2px solid #27272A !important;
            margin: 30px 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom: 25px; border-left: 6px solid #E11D48; padding-left: 15px; padding-top: 5px; padding-bottom: 5px;">
            <h2 style="margin: 0; color: #FAFAFA; font-size: 32px; font-weight: 700;">{title}</h2>
            <p style="color: #A1A1AA; font-size: 15px; font-weight: 600; font-family: 'Inter'; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: url('https://www.transparenttextures.com/patterns/stardust.png'), #09090B; border: 2px solid #27272A; padding: 40px 30px; 
            box-shadow: 8px 8px 0px rgba(0,0,0,0.8); margin-bottom: 40px; position: relative; overflow: hidden; border-bottom: 6px solid #E11D48;">
            <div style="position: absolute; right: 20px; bottom: -20px; font-size: 140px; opacity: 0.05;">{icon}</div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="font-size: 56px; text-shadow: 3px 3px 0px #000;">{icon}</div>
                <div>
                    <h1 style="font-size: 52px; margin: 0; color: #FAFAFA; text-shadow: 2px 2px 0px #000;">{title}</h1>
                    <p style="color: #E11D48; font-size: 16px; font-weight: 700; margin-top: 4px; letter-spacing: 2px; text-transform: uppercase; font-family: 'Oswald';">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#E11D48", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 20, 'color': '#FAFAFA', 'family': 'Oswald'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 2, 'tickcolor': "#3F3F46"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#09090B",
            'borderwidth': 0,
            'steps': [{'range': [0, max_val], 'color': '#18181B'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FAFAFA", 'family': "Oswald"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #A1A1AA; font-family: monospace; font-size: 14px; margin-bottom: 2px;">'
                        f'<span style="color: #52525B;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-10:]])
    placeholder.markdown(f"""
        <div style="background: #050505; border: 2px solid #27272A; padding: 20px; box-shadow: 4px 4px 0px #000;">
            <div style="font-family: 'Oswald'; font-size: 18px; color: #FAFAFA; margin-bottom: 12px; letter-spacing: 1px; border-bottom: 1px solid #27272A; padding-bottom: 8px;">TARGET ACQUISITION LOG</div>
            {log_html if logs else '<div style="color: #52525B; font-family: monospace; font-size: 14px;">Awaiting feed...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#0284C7"):
    st.markdown(f"""
        <div style="background: #0A0A0A; border: 2px solid #27272A; padding: 18px; margin-bottom: 15px; box-shadow: 4px 4px 0px {clr};">
            <div style="font-family:'Oswald'; font-size:14px; color:{clr}; text-transform:uppercase; letter-spacing: 1px; margin-bottom: 4px;">{label}</div>
            <div style="font-size:20px; font-weight:600; color:#FAFAFA; font-family:'Inter';">{text}</div>
        </div>
    """, unsafe_allow_html=True)
