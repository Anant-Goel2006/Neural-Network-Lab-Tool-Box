import streamlit as st
import time
import plotly.graph_objects as go

def inject_global_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;500;700&family=Inter:wght@400;600&display=swap');
        
        .stApp {
            background-color: #040014;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 30%, rgba(0, 240, 255, 0.12) 0%, transparent 40%),
                url('https://www.transparenttextures.com/patterns/stardust.png');
            background-attachment: fixed;
            color: #E4E4E7;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── COSMIC GRAPHIC NOVEL PANELS ──── */
        .premium-card {
            background: rgba(10, 10, 20, 0.85);
            backdrop-filter: blur(8px);
            border: 3px solid #1e1b4b;
            box-shadow: 6px 6px 0px #8b5cf6;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.2s;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .premium-card:hover {
            transform: translate(-3px, -3px);
            box-shadow: 9px 9px 0px #00f0ff;
            border-color: #2e1065;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #FAFAFA;
        }

        /* ──── COMIC BUTTONS ──── */
        .stButton > button {
            background-color: rgba(10, 10, 20, 0.9) !important;
            border: 3px solid #3b0764 !important;
            border-radius: 0px !important;
            color: #FAFAFA !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            transition: all 0.1s ease-in-out !important;
            padding: 10px 24px !important;
            box-shadow: 5px 5px 0px #8b5cf6 !important;
        }
        .stButton > button:hover {
            border-color: #00f0ff !important;
            box-shadow: 7px 7px 0px #00f0ff !important;
            transform: translate(-2px, -2px) !important;
            color: #00f0ff !important;
        }
        .stButton > button:active {
            transform: translate(3px, 3px) !important;
            box-shadow: 2px 2px 0px #00f0ff !important;
        }

        /* ──── HEADER MENU BAR ──── */
        header[data-testid="stHeader"] {
            background-color: #050505 !important;
            border-bottom: 3px solid #8b5cf6 !important;
            box-shadow: none !important;
        }
        header[data-testid="stHeader"] * {
            color: #00f0ff !important;
            font-family: 'Oswald', sans-serif !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: #040014 !important;
            border-right: 3px solid #2e1065;
        }
        
        hr {
            border-top: 3px dashed #3b0764 !important;
            margin: 30px 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:30px; border-left: 8px solid #8b5cf6; padding-left:15px; position:relative; word-wrap: break-word;">
            <div style="position:absolute; top:0; left:0; width:100%; height:3px; background: linear-gradient(90deg, #00f0ff, transparent);"></div>
            <h2 style="margin:0; color:#FAFAFA; font-size:32px; font-weight:700; text-shadow: 3px 3px 0px #000; text-transform:uppercase;">{title}</h2>
            <p style="color:#00f0ff; font-size:16px; font-weight:600; letter-spacing:2px; margin-top:4px; font-family:'Inter'; text-transform:uppercase;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b; padding:40px 30px; 
            box-shadow: 8px 8px 0px #8b5cf6; margin-bottom: 30px; position: relative; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <div style="font-size: 72px; text-shadow: 4px 4px 0px #8b5cf6;">{icon}</div>
                <div style="flex:1; min-width: 250px;">
                    <h1 style="font-size: 56px; margin: 0; color: #FAFAFA; font-weight: 700; letter-spacing: 3px; text-shadow: 4px 4px 0px #8b5cf6; line-height: 1.1;">{title}</h1>
                    <p style="color:#00f0ff; font-size:18px; font-family:'Oswald'; font-weight:600; margin-top:8px; letter-spacing:2px; background:#040014; display:inline-block; padding: 4px 10px; border: 2px solid #1e1b4b;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#00f0ff", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 20, 'color': '#FAFAFA', 'family': 'Oswald'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 3, 'tickcolor': "#3b0764"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#040014",
            'borderwidth': 0,
            'steps': [{'range': [0, max_val], 'color': '#1e1b4b'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FAFAFA", 'family': "Oswald"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #A1A1AA; font-family: monospace; font-size: 14px; margin-bottom: 4px; overflow-wrap: break-word;">'
                        f'<span style="color: #00f0ff; font-weight: bold;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-10:]])
    placeholder.markdown(f"""
        <div style="background: rgba(10,10,20,0.9); border: 3px solid #3b0764; padding: 20px; box-shadow: 6px 6px 0px #000; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="font-family: 'Oswald'; font-size: 20px; color: #FAFAFA; margin-bottom: 12px; letter-spacing: 2px; border-bottom: 2px solid #2e1065; padding-bottom: 8px;">TARGET ACQUISITION LOG</div>
            {log_html if logs else '<div style="color: #52525B; font-family: monospace; font-size: 14px;">Awaiting feed...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#00f0ff"):
    st.markdown(f"""
        <div style="background: rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b; border-left: 8px solid {clr}; padding: 20px; margin-bottom: 24px; box-shadow: 6px 6px 0px #8b5cf6; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="font-family:'Oswald', sans-serif; font-size:18px; font-weight:700; color:{clr}; letter-spacing: 2px; margin-bottom: 8px; text-transform: uppercase;">
                {label} // NLP OVERRIDE
            </div>
            <div style="font-size:16px; color:#FAFAFA; font-family:'Inter', sans-serif; line-height: 1.6;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
