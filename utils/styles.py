import streamlit as st
import time
import plotly.graph_objects as go

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── IRON MAN & BATMAN COMIC: PROFESSIONAL GRAPHICS ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp {
            background-color: #0B0E14; /* Deep Gotham/Space Black */
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            background-image: 
                /* Stars */
                radial-gradient(1.5px 1.5px at 20px 30px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 80px 70px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 150px 160px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 250px 40px, #ffffff, rgba(0,0,0,0)),
                /* Neural Net Mesh SVG */
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25'%3E%3Cdefs%3E%3Cpattern id='brain' width='150' height='150' patternUnits='userSpaceOnUse'%3E%3Ccircle cx='30' cy='30' r='2' fill='%2338bdf8' opacity='0.6'/%3E%3Ccircle cx='110' cy='60' r='3' fill='%23ef4444' opacity='0.7'/%3E%3Ccircle cx='60' cy='120' r='2.5' fill='%23facc15' opacity='0.6'/%3E%3Cpath d='M30 30 L110 60 L60 120 Z' fill='none' stroke='rgba(255,255,255,0.07)' stroke-width='1'/%3E%3Cpath d='M-10 -10 L30 30' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1'/%3E%3Cpath d='M110 60 L180 30' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23brain)'/%3E%3C/svg%3E");
            background-attachment: fixed;
        }

        /* Ambient glowing overlay for that Iron-Man core / cosmic feel */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at 100% 0%, rgba(229, 9, 20, 0.15) 0%, transparent 60%),
                        radial-gradient(circle at 0% 100%, rgba(56, 189, 248, 0.1) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        /* ──── PROFESSIONAL COMIC CARDS ──── */
        .premium-card {
            background: #141923; /* Matte Dark panel */
            border: 3px solid #000000; /* Hard comic borders */
            border-radius: 2px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 6px 6px 0px rgba(0,0,0,0.9); /* Sharp comic shadows */
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
        }
        
        .premium-card:hover {
            transform: translate(-4px, -4px);
            box-shadow: 10px 10px 0px #DC2626; /* Stark Red comic jump */
            border-color: #000000;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase;
        }

        /* ──── COMIC ACTION BUTTONS (STARK & WAYNE TECH) ──── */
        .stButton > button {
            background: #DC2626 !important; /* Bold Marvel Red */
            border: 3px solid #000000 !important;
            border-radius: 2px !important;
            color: #ffffff !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 22px !important;
            text-transform: uppercase !important;
            box-shadow: 4px 4px 0px #000000 !important;
            padding: 10px 24px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
        }
        .stButton > button:hover {
            background: #FACC15 !important; /* Batman/Iron-Man Yellow */
            color: #000000 !important;
            transform: translate(-3px, -3px) !important;
            box-shadow: 7px 7px 0px #000000 !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: #0F1219 !important;
            border-right: 3px solid #000000;
        }

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 42px !important;
            color: #FACC15 !important;
            text-shadow: 2px 2px 0px #000000, -1px -1px 0px #000000, 1px -1px 0px #000000, -1px 1px 0px #000000; /* Comic outline */
        }
        </style>
    """, unsafe_allow_html=True)


def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:35px; position: relative; z-index: 10; transform: skewX(-3deg);">
            <div style="display:inline-block; border-bottom: 5px solid #FACC15; padding-bottom: 5px; padding-right: 20px; box-shadow: 3px 3px 0px rgba(0,0,0,0.4); background: rgba(0,0,0,0.3); padding-left: 10px;">
                <h2 style="margin:0; font-size:38px; line-height: 1; text-shadow: 2px 2px 0px #000;">{title}</h2>
            </div>
            <p style="color:#D1D5DB; font-size:16px; font-weight:700; margin-top:12px; margin-left:10px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 2px; text-shadow: 1px 1px 0px #000000;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #141923 0%, #0B0E14 100%); border: 4px solid #000000; padding:40px; 
            margin-bottom: 45px; box-shadow: 12px 12px 0px #DC2626; position:relative; overflow: hidden; transform: skewX(-1deg);">
            <div style="position: absolute; right: -20px; top: -50px; opacity: 0.1; font-size: 250px; font-family: 'Bebas Neue'; transform: rotate(15deg);">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 30px; position: relative; z-index: 2;">
                <div style="font-size: 70px; filter: drop-shadow(4px 4px 0px #000000);">{icon}</div>
                <div style="flex:1;">
                    <h1 style="font-size: 64px; margin: 0; line-height:1; color: #FFFFFF; text-shadow: 3px 3px 0px #000000, -2px -2px 0px #000, 2px -2px 0px #000, -2px 2px 0px #000;">{title}</h1>
                    <p style="color:#FACC15; font-size:22px; font-family:'Inter', sans-serif; font-weight: 700; letter-spacing:2px; margin-top:15px; text-transform: uppercase; text-shadow: 2px 2px 0px #000000;">
                        {sub}
                    </p>
                </div>
            </div>
            <div style="position:absolute; bottom:0; left:0; width:100%; height:8px; background:#FACC15; border-top:2px solid #000000;"></div>
        </div>
    """, unsafe_allow_html=True)


def speedometer(val, max_val, title, color="#DC2626", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 24, 'color': '#FFFFFF', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 3, 'tickcolor': "#000000"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.6)",
            'borderwidth': 3,
            'bordercolor': "#000000",
            'steps': [{'range': [0, max_val], 'color': 'rgba(250, 204, 21, 0.15)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #E2E8F0; font-family: \'JetBrains Mono\', monospace; font-size: 14px; margin-bottom: 6px; border-bottom: 1px dashed #334155; padding-bottom:4px;">'
                        f'<span style="color: #FACC15; font-weight: bold; text-shadow: 1px 1px 0px #000;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: #0B0E14; border: 3px solid #000000; padding: 25px; box-shadow: 6px 6px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 26px; color: #38BDF8; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 0px #000000;">BATCOMPUTER / JARVIS UPLINK</div>
            {log_html if logs else '<div style="color: #94A3B8; font-family: \'JetBrains Mono\', monospace;">SYS.STANDBY...</div>'}
            <div style="position: absolute; top:0; right:0; padding: 10px; opacity: 0.5;">
                <div style="width:10px; height:10px; background:#EF4444; border-radius:50%; border:2px solid #000;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_nlp_insight(text, label, clr="#38BDF8"):
    st.markdown(f"""
        <div style="background: #141923; border: 3px solid #000000; border-left: 12px solid {clr}; padding: 30px; margin-bottom: 35px; box-shadow: 8px 8px 0px #000000; position:relative; transform: skewX(1deg);">
            <div style="font-family:'Bebas Neue', cursive; font-size:32px; color:#FFFFFF; letter-spacing: 2px; margin-bottom:15px; text-shadow: 2px 2px 0px #000000;">
                {label}
            </div>
            <div style="font-size:16px; color:#0e0e0e; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 700; background: #FFFFFF; padding: 20px; border: 3px solid #000000; box-shadow: inset 2px 2px 0px rgba(0,0,0,0.1);">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
