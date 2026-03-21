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

        /* ──── COMIC SPACE PANELS ──── */
        .premium-card {
            background: rgba(10, 10, 20, 0.85);
            backdrop-filter: blur(8px);
            border: 3px solid #1e1b4b;
            box-shadow: 6px 6px 0px #8b5cf6;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.2s ease;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .premium-card:hover {
            transform: translate(-3px, -3px);
            box-shadow: 9px 9px 0px #00f0ff;
            border-color: #00f0ff;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #FAFAFA;
        }

        /* ──── SLEEK GLOW BUTTONS ──── */
        .stButton > button {
            background: rgba(10, 10, 20, 0.8) !important;
            border: 1px solid rgba(139, 92, 246, 0.5) !important;
            border-radius: 8px !important;
            color: #FAFAFA !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            font-family: 'Oswald', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            transition: all 0.2s ease !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
        }
        .stButton > button:hover {
            border-color: #00f0ff !important;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
            transform: translateY(-2px) !important;
            color: #00f0ff !important;
        }
        .stButton > button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 0 5px rgba(0, 240, 255, 0.4) !important;
        }

        /* ──── HEADER MENU BAR ──── */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            background-image: linear-gradient(to bottom, rgba(4,0,20,0.8), transparent) !important;
            box-shadow: none !important;
        }
        header[data-testid="stHeader"] * {
            color: #E4E4E7 !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(4, 0, 20, 0.8) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        hr {
            border-top: 1px dashed rgba(139, 92, 246, 0.3) !important;
            margin: 30px 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:30px; position:relative; word-wrap: break-word; text-align: center;">
            <h2 style="margin:0; color:#FAFAFA; font-size:36px; font-weight:700; letter-spacing: 2px; text-transform:uppercase; filter: drop-shadow(0 0 10px rgba(139,92,246,0.3));">{title}</h2>
            <p style="color:#00f0ff; font-size:16px; font-weight:400; letter-spacing:4px; margin-top:8px; font-family:'Inter'; text-transform:uppercase;">{subtitle}</p>
            <div style="width: 60px; height: 3px; background: linear-gradient(90deg, transparent, #8b5cf6, #00f0ff, transparent); margin: 15px auto 0;"></div>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 4px solid #1e1b4b; border-bottom: 6px solid #8b5cf6; border-radius: 12px; padding:40px 30px; 
            box-shadow: 8px 8px 0px #00f0ff; margin-bottom: 30px; position: relative; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <div style="font-size: 72px; text-shadow: 5px 5px 0px #8b5cf6;">{icon}</div>
                <div style="flex:1; min-width: 250px;">
                    <h1 style="font-size: 48px; margin: 0; color: #FAFAFA; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; text-shadow: 3px 3px 0px #8b5cf6, -2px -2px 0px #00f0ff;">{title}</h1>
                    <p style="color:#E4E4E7; font-size:16px; font-family:'Inter'; font-weight:700; margin-top:12px; letter-spacing:4px; text-transform: uppercase; background:#040014; display:inline-block; padding: 5px 12px; border: 2px solid #1e1b4b;">
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
        title = {'text': title.upper(), 'font': {'size': 18, 'color': '#FAFAFA', 'family': 'Oswald'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 2, 'tickcolor': "rgba(139,92,246,0.5)"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 0,
            'steps': [{'range': [0, max_val], 'color': 'rgba(10,10,20,0.8)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FAFAFA", 'family': "Oswald"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #A1A1AA; font-family: monospace; font-size: 14px; margin-bottom: 4px; overflow-wrap: break-word;">'
                        f'<span style="color: #00f0ff;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-10:]])
    placeholder.markdown(f"""
        <div style="background: rgba(10,10,20,0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b; border-radius: 8px; padding: 20px; box-shadow: 6px 6px 0px #8b5cf6; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="font-family: 'Oswald'; font-size: 18px; color: #FAFAFA; margin-bottom: 12px; letter-spacing: 2px; border-bottom: 2px solid #1e1b4b; padding-bottom: 8px;">TARGET ACQUISITION LOG</div>
            {log_html if logs else '<div style="color: #52525B; font-family: monospace; font-size: 14px;">Awaiting feed...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#00f0ff"):
    st.markdown(f"""
        <div style="background: rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b; border-left: 6px solid {clr}; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 6px 6px 0px {clr}; word-wrap: break-word; overflow-wrap: break-word;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="font-size: 20px; text-shadow: 2px 2px 0px {clr};">🤖</div>
                <div style="font-family:'Oswald', sans-serif; font-size:16px; font-weight:700; color:{clr}; letter-spacing: 2px; text-transform: uppercase;">
                    {label} // NLP OVERRIDE
                </div>
            </div>
            <div style="font-size:15px; color:#E4E4E7; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 600;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
