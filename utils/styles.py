import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── DAYLIGHT COMIC PRINT: MAXIMUM LEGIBILITY & LAYOUT FIXES ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #0f172a; /* Deep comic ink for body text visibility */
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── COMIC PAPER CARDS ──── */
        .premium-card {
            background: #ffffff; /* Bright comic paper white */
            border: 3px solid #000000; /* Thick black comic ink border */
            border-bottom: 5px solid #000000; /* Grounding the comic panel */
            border-radius: 2px; 
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 6px 6px 0px rgba(0, 0, 0, 1); /* Hard black offset shadow */
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
            word-wrap: break-word; /* Ensure text never overflows */
        }
        
        .premium-card:hover {
            transform: translate(-3px, -3px);
            box-shadow: 9px 9px 0px #3B82F6; /* Comic Blue shadow hover lift */
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #000000 !important; /* Pitch black headers for light theme */
            letter-spacing: 2px !important;
            text-transform: uppercase;
            word-wrap: break-word;
        }

        /* ──── YELLOW ACTION BUTTONS ──── */
        .stButton > button {
            background: #FACC15 !important; /* Bright Action Yellow */
            border: 3px solid #000000 !important;
            border-radius: 2px !important;
            color: #000000 !important; /* Black text for visibility */
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 22px !important;
            text-transform: uppercase !important;
            box-shadow: 4px 4px 0px #000000 !important;
            padding: 10px 20px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
            white-space: normal !important; 
            height: auto !important;
        }
        .stButton > button:hover {
            background: #DC2626 !important; /* Bold Red on Hover */
            color: #ffffff !important;
            transform: translate(-2px, -2px) !important;
            box-shadow: 6px 6px 0px #000000 !important; 
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important; /* Clean gray paper sidebar */
            border-right: 4px solid #000000;
        }
        
        /* Sidebar Text color override for light background */
        [data-testid="stSidebar"] * {
            color: #0f172a !important; /* Force all text to be dark grey/black in sidebar */
        }
        

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 40px !important; 
            color: #DC2626 !important; /* Bold Marvel Red */
            word-wrap: break-word;
            text-shadow: 2px 2px 0px #000000; /* Single tight drop-shadow so it isn't completely blown out */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT LIVE DAYLIGHT COMIC BACKGROUND ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('light-comic-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'light-comic-bg';
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
                const particleCount = 110;
                
                // Bright Comic ink colors
                const colors = ['#DC2626', '#3B82F6', '#FACC15', '#000000'];
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 1.5,
                        vy: (Math.random() - 0.5) * 1.5,
                        radius: Math.random() * 2.5 + 1.2,
                        color: colors[Math.floor(Math.random() * colors.length)]
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#F1F5F9'; // Light paper backdrop 
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
                            
                            if(dist < 130) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                ctx.lineTo(p2.x, p2.y); 
                                
                                let alpha = (1 - (dist / 130)) * 0.5;
                                ctx.globalAlpha = alpha;
                                ctx.strokeStyle = '#000000'; // Black comic ink connection lines
                                ctx.lineWidth = 1;
                                ctx.stroke();
                                ctx.globalAlpha = 1.0;
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
                
                // Cleanup old dark backgrounds
                const oldMarvel = parentDoc.getElementById('marvel-live-bg');
                if(oldMarvel) oldMarvel.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    # Maximum contrast blocky text sections with clear word wrapping
    st.markdown(f"""
        <div style="margin-bottom:30px; position: relative; z-index: 10; transform: skewX(-2deg);">
            <div style="display:inline-block; border-bottom: 4px solid #000000; padding-bottom: 5px; padding-right: 25px; box-shadow: 4px 4px 0px #FACC15; background: #ffffff; padding-left: 15px; border-left: 5px solid #000000; border-top: 3px solid #000; border-right: 3px solid #000; max-width: 100%;">
                <h2 style="margin:0; font-size:32px; line-height: 1.1; color: #000000; text-shadow: 2px 2px 0px #e2e8f0; word-wrap: break-word;">{title}</h2>
            </div>
            <p style="color:#1e293b; font-size:15px; font-weight:800; margin-top:12px; margin-left:15px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 1.5px; word-wrap: break-word; text-shadow: 1px 1px 0px #ffffff;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    # Complete comic block header, entirely safe on Light mode
    st.markdown(f"""
        <div style="background: #F8FAFC; border: 4px solid #000000; padding:30px; 
            margin-bottom: 35px; box-shadow: 10px 10px 0px #3B82F6; position:relative; overflow: hidden; transform: skewX(-1deg);">
            <div style="position: absolute; right: -10px; top: -10px; opacity: 0.1; font-size: 150px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #000000;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 2; flex-wrap: wrap;">
                <div style="font-size: 60px; filter: drop-shadow(4px 4px 0px #000000);">{icon}</div>
                <div style="flex:1; min-width: 200px;">
                    <h1 style="font-size: 50px; margin: 0; line-height:1.1; color: #000000; text-shadow: 3px 3px 0px #FACC15, -1px -1px 0px #000000, 1px -1px 0px #000, -1px 1px 0px #000; word-wrap: break-word;">{title}</h1>
                    <p style="color:#ffffff; font-size:18px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:2px; margin-top:12px; text-transform: uppercase; text-shadow: none; background: #DC2626; display: inline-block; padding: 5px 12px; border: 3px solid #000000; box-shadow: 3px 3px 0px #000000; word-wrap: break-word;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#3B82F6", height=200):
    # Dark font properties to guarantee visibility over white paper wrapper
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 20, 'color': '#000000', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 3, 'tickcolor': "#000000"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#e2e8f0",
            'borderwidth': 3,
            'bordercolor': "#000000",
            'steps': [{'range': [0, max_val], 'color': 'rgba(0, 0, 0, 0.05)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#000000", 'family': "Bebas Neue"}, height=height, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #000000; font-family: \'JetBrains Mono\', monospace; font-size: 15px; margin-bottom: 6px; border-bottom: 1px dashed #94a3b8; padding-bottom:4px; word-wrap: break-word; font-weight:600;">'
                        f'<span style="color: #DC2626; font-weight: 900;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: #ffffff; border: 4px solid #000000; padding: 20px; box-shadow: 6px 6px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 26px; color: #000000; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: #FACC15; padding: 3px 12px; font-weight: 700; border: 3px solid #000000; box-shadow: 2px 2px 0px #000000;">SYSTEM LOG</div>
            {log_html if logs else '<div style="color: #475569; font-family: \'JetBrains Mono\', monospace; font-weight:bold;">AWAITING SYSTEM BOOT...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#FACC15"):
    st.markdown(f"""
        <div style="background: #ffffff; border: 4px solid #000000; border-left: 12px solid {clr}; padding: 25px; margin-bottom: 25px; box-shadow: 8px 8px 0px #000000; position:relative; transform: skewX(1deg);">
            <div style="font-family:'Bebas Neue', cursive; font-size:28px; color:#000000; letter-spacing: 2px; margin-bottom:12px; word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size:15px; color:#0f172a; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 800; background: #F8FAFC; padding: 15px; border: 3px solid #000000; box-shadow: inset 2px 2px 0px rgba(0,0,0,0.1); word-wrap: break-word;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
