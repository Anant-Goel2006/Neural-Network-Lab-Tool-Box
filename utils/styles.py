import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── DC BATMAN COMIC: COLOR GRADING & LAYOUT FIXES ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── COMIC CARDS (BATCAVE STEALTH) ──── */
        .premium-card {
            background: rgba(10, 12, 16, 0.9); /* Deep Gotham Black */
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 3px solid #000000; 
            border-bottom: 3px solid #FACC15; /* Signal Yellow grounding */
            border-radius: 4px; 
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 6px 6px 0px rgba(0, 0, 0, 0.9); 
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
            word-wrap: break-word; /* Prevents overflow */
        }
        
        .premium-card:hover {
            transform: translate(-3px, -3px);
            box-shadow: 9px 9px 0px #FACC15; /* Yellow block shadow lift */
            border-color: #000000;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase;
            word-wrap: break-word;
        }

        /* ──── BAT-BUTTONS ──── */
        .stButton > button {
            background: #000000 !important; /* Pitch Black */
            border: 3px solid #FACC15 !important;
            border-radius: 2px !important;
            color: #FACC15 !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 20px !important;
            text-transform: uppercase !important;
            box-shadow: 5px 5px 0px #FACC15 !important;
            padding: 8px 20px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
            white-space: normal !important; 
            height: auto !important;
        }
        .stButton > button:hover {
            background: #FACC15 !important; /* Striking Yellow */
            color: #000000 !important;
            border-color: #000000 !important;
            transform: translate(-3px, -3px) !important;
            box-shadow: 8px 8px 0px #000000 !important; /* Dark drop */
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(5, 7, 10, 0.95) !important;
            backdrop-filter: blur(15px) !important;
            border-right: 3px solid #FACC15;
        }

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 38px !important; 
            color: #FACC15 !important;
            word-wrap: break-word;
            text-shadow: 2px 2px 0px #000000, -1px -1px 0px #000000, 1px -1px 0px #000000, -1px 1px 0px #000000; 
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT LIVE BATMAN GRADING NEURONS BACKGROUND ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('bat-live-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'bat-live-bg';
                canvas.style.position = 'fixed';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100vw';
                canvas.style.height = '100vh';
                canvas.style.zIndex = '-1'; 
                canvas.style.pointerEvents = 'none';
                
                parentDoc.body.insertBefore(canvas, parentDoc.body.firstChild);
                
                const ctx = canvas.getContext('2d');
                let width = canvas.width = parentDoc.defaultView.innerWidth;
                let height = canvas.height = parentDoc.defaultView.innerHeight;
                
                parentDoc.defaultView.addEventListener('resize', () => {
                    width = canvas.width = parentDoc.defaultView.innerWidth;
                    height = canvas.height = parentDoc.defaultView.innerHeight;
                });

                const particles = [];
                const particleCount = 100;
                
                // DC Batman Palette: Graphite, Slate, Gunmetal, Signall Yellow
                const colors = ['#334155', '#475569', '#1e293b', '#FACC15', '#f59e0b'];
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 1.5,
                        vy: (Math.random() - 0.5) * 1.5,
                        radius: Math.random() * 3 + 1.2,
                        color: colors[Math.floor(Math.random() * colors.length)]
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#050608'; // Pitch black Gotham night 
                    ctx.fillRect(0, 0, width, height);
                    
                    for(let i=0; i<particleCount; i++) {
                        let p = particles[i];
                        p.x += p.vx;
                        p.y += p.vy;
                        
                        if(p.x < 0 || p.x > width) p.vx *= -1;
                        if(p.y < 0 || p.y > height) p.vy *= -1;
                        
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                        ctx.fillStyle = p.color; 
                        ctx.fill();
                        
                        for(let j=i+1; j<particleCount; j++) {
                            let p2 = particles[j];
                            let dx = p.x - p2.x;
                            let dy = p.y - p2.y;
                            let dist = Math.sqrt(dx*dx + dy*dy);
                            
                            if(dist < 140) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                ctx.lineTo(p2.x, p2.y); 
                                
                                let alpha = (1 - (dist / 140)) * 0.5;
                                ctx.globalAlpha = alpha;
                                ctx.strokeStyle = p.color; 
                                ctx.lineWidth = 1;
                                ctx.stroke();
                                ctx.globalAlpha = 1.0;
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
                
                // Cleanup old backgrounds
                const oldComic = parentDoc.getElementById('comic-live-bg');
                if(oldComic) oldComic.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:30px; position: relative; z-index: 10; transform: skewX(-2deg);">
            <div style="display:inline-block; border-bottom: 4px solid #FACC15; padding-bottom: 5px; padding-right: 15px; box-shadow: 4px 4px 0px #000000; background: rgba(0,0,0,0.9); padding-left: 10px; border-left: 5px solid #FACC15; max-width: 100%;">
                <h2 style="margin:0; font-size:28px; line-height: 1.1; text-shadow: 2px 2px 0px #000; word-wrap: break-word;">{title}</h2>
            </div>
            <p style="color:#94a3b8; font-size:14px; font-weight:700; margin-top:10px; margin-left:10px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 1px; text-shadow: 1px 1px 0px #000000; word-wrap: break-word;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.95); backdrop-filter: blur(8px); border: 4px solid #000000; padding:25px; 
            margin-bottom: 35px; box-shadow: 8px 8px 0px #FACC15; position:relative; overflow: hidden; transform: skewX(-1deg);">
            <div style="position: absolute; right: -10px; top: -10px; opacity: 0.1; font-size: 150px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #FACC15;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 2; flex-wrap: wrap;">
                <div style="font-size: 55px; filter: drop-shadow(3px 3px 0px #000000);">{icon}</div>
                <div style="flex:1; min-width: 200px;">
                    <h1 style="font-size: 42px; margin: 0; line-height:1.1; color: #FFFFFF; text-shadow: 3px 3px 0px #000000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000; word-wrap: break-word;">{title}</h1>
                    <p style="color:#000000; font-size:18px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:2px; margin-top:10px; text-transform: uppercase; text-shadow: none; background: #FACC15; display: inline-block; padding: 4px 10px; border: 2px solid #000; word-wrap: break-word;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#FACC15", height=200):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 20, 'color': '#FFFFFF', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 3, 'tickcolor': "#000000"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.8)",
            'borderwidth': 3,
            'bordercolor': "#000000",
            'steps': [{'range': [0, max_val], 'color': 'rgba(250, 204, 21, 0.15)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #cbd5e1; font-family: \'JetBrains Mono\', monospace; font-size: 14px; margin-bottom: 6px; border-bottom: 1px dashed rgba(250,204,21,0.3); padding-bottom:4px; word-wrap: break-word;">'
                        f'<span style="color: #FACC15; font-weight: bold; text-shadow: 1px 1px 0px #000;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.95); border: 3px solid #000000; padding: 20px; box-shadow: 6px 6px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 24px; color: #000000; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: #FACC15; padding: 2px 10px; font-weight: 700; border: 2px solid #000;">BATCOMPUTER UPLINK</div>
            {log_html if logs else '<div style="color: #475569; font-family: \'JetBrains Mono\', monospace;">SYS.STANDBY...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#FACC15"):
    st.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.9); border: 3px solid #000000; border-left: 10px solid {clr}; padding: 25px; margin-bottom: 25px; box-shadow: 8px 8px 0px #000000; position:relative; transform: skewX(1deg);">
            <div style="font-family:'Bebas Neue', cursive; font-size:26px; color:#FFFFFF; letter-spacing: 2px; margin-bottom:12px; text-shadow: 2px 2px 0px #000000; word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size:15px; color:#0f172a; font-family:'Inter', sans-serif; line-height: 1.5; font-weight: 800; background: #FFFFFF; padding: 15px; border: 2px solid #000000; box-shadow: inset 2px 2px 0px rgba(0,0,0,0.15); word-wrap: break-word;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
