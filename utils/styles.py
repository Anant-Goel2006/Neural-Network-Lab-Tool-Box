import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── BIOLOGICAL CORTEX BACKGROUND + COMIC TEXT THEME ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── NEURAL ORGANIC CARDS + COMIC BORDERS ──── */
        .premium-card {
            background: rgba(17, 12, 34, 0.75); /* Deep neural violet */
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 3px solid #000000; /* Restored Comic Border */
            border-radius: 4px; /* Restored sharp edges */
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 8px 8px 0px rgba(0, 0, 0, 0.9); /* Restored Comic shadow */
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
        }
        
        .premium-card:hover {
            transform: translate(-4px, -4px);
            box-shadow: 12px 12px 0px #d946ef; /* Magenta comic shadow */
            border-color: #000000;
        }

        /* RESTORED BEBAS NEUE TEXT THEME */
        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase;
        }

        /* ──── SYNAPTIC PULSE BUTTONS (COMIC TEXT) ──── */
        .stButton > button {
            background: #d946ef !important; /* Bold Neural Magenta */
            border: 3px solid #000000 !important;
            border-radius: 2px !important;
            color: #ffffff !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 24px !important;
            text-transform: uppercase !important;
            box-shadow: 5px 5px 0px #000000 !important;
            padding: 10px 24px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
        }
        .stButton > button:hover {
            background: #06b6d4 !important; /* Cyan */
            color: #000000 !important;
            border-color: #000000 !important;
            transform: translate(-3px, -3px) !important;
            box-shadow: 8px 8px 0px #000000 !important;
        }

        /* ──── CORTEX SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(6, 4, 15, 0.95) !important;
            backdrop-filter: blur(25px) !important;
            border-right: 3px solid #000000;
        }

        /* ──── METRICS (COMIC TEXT + BIOLUMINESCENT COLORS) ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 46px !important;
            color: #67e8f9 !important; /* Cyan bio-glow */
            text-shadow: 2px 2px 0px #000000, -1px -1px 0px #000000, 1px -1px 0px #000000, -1px 1px 0px #000000; /* Comic outline */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT LIVE ANIMATED HTML5 CANVAS FOR JS NEURAL CORTEX BACKGROUND ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('bio-cortex-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'bio-cortex-bg';
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
                const particleCount = 120;
                
                // Bioluminescent Synapse Theme (Magenta, Cyan, Violet)
                const colors = ['#d946ef', '#06b6d4', '#8b5cf6'];
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 0.8,
                        vy: (Math.random() - 0.5) * 0.8,
                        radius: Math.random() * 3 + 0.5,
                        color: colors[Math.floor(Math.random() * colors.length)],
                        pulse: Math.random() * Math.PI * 2
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#06040f'; 
                    ctx.fillRect(0, 0, width, height);
                    
                    for(let i=0; i<particleCount; i++) {
                        let p = particles[i];
                        p.x += p.vx;
                        p.y += p.vy;
                        p.pulse += 0.05; 
                        
                        if(p.x < 0 || p.x > width) p.vx *= -1;
                        if(p.y < 0 || p.y > height) p.vy *= -1;
                        
                        let currentRadius = p.radius + Math.sin(p.pulse) * 1.5;
                        if(currentRadius < 0.1) currentRadius = 0.1;
                        
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, currentRadius, 0, Math.PI * 2);
                        ctx.fillStyle = p.color; 
                        ctx.shadowBlur = 15;
                        ctx.shadowColor = p.color;
                        ctx.fill();
                        
                        for(let j=i+1; j<particleCount; j++) {
                            let p2 = particles[j];
                            let dx = p.x - p2.x;
                            let dy = p.y - p2.y;
                            let dist = Math.sqrt(dx*dx + dy*dy);
                            
                            if(dist < 140) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                
                                let cx = (p.x + p2.x) / 2 + (Math.random() - 0.5) * 5;
                                let cy = (p.y + p2.y) / 2 + (Math.random() - 0.5) * 5;
                                
                                ctx.quadraticCurveTo(cx, cy, p2.x, p2.y);
                                
                                let alpha = (1 - (dist / 140)) * 0.6;
                                ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`; 
                                ctx.lineWidth = 1;
                                ctx.shadowBlur = 0; 
                                ctx.stroke();
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
                
                const oldBat = parentDoc.getElementById('batman-live-bg');
                if(oldBat) oldBat.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    # Restored heavy comic headers
    st.markdown(f"""
        <div style="margin-bottom:35px; position: relative; z-index: 10; transform: skewX(-4deg);">
            <div style="display:inline-block; border-bottom: 5px solid #d946ef; padding-bottom: 5px; padding-right: 20px; box-shadow: 4px 4px 0px #000000; background: rgba(0,0,0,0.8); padding-left: 15px; border-left: 5px solid #d946ef;">
                <h2 style="margin:0; font-size:40px; line-height: 1; text-shadow: 2px 2px 0px #000;">{title}</h2>
            </div>
            <p style="color:#e2e8f0; font-size:16px; font-weight:700; margin-top:12px; margin-left:15px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 2px; text-shadow: 1px 1px 0px #000000;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    # Restored heavy comic headers
    st.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.9); backdrop-filter: blur(10px); border: 4px solid #000000; padding:40px; 
            margin-bottom: 45px; box-shadow: 12px 12px 0px #06b6d4; position:relative; overflow: hidden; transform: skewX(-2deg);">
            <div style="position: absolute; right: -20px; top: -50px; opacity: 0.1; font-size: 250px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #06b6d4;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 30px; position: relative; z-index: 2;">
                <div style="font-size: 75px; filter: drop-shadow(4px 4px 0px #000000);">{icon}</div>
                <div style="flex:1;">
                    <h1 style="font-size: 68px; margin: 0; line-height:1; color: #FFFFFF; text-shadow: 4px 4px 0px #000000, -2px -2px 0px #000, 2px -2px 0px #000, -2px 2px 0px #000;">{title}</h1>
                    <p style="color:#d946ef; font-size:24px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:3px; margin-top:15px; text-transform: uppercase; text-shadow: 2px 2px 0px #000000; background: #000; display: inline-block; padding: 4px 12px;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#06b6d4", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 26, 'color': '#FFFFFF', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 4, 'tickcolor': "#000000"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.8)",
            'borderwidth': 4,
            'bordercolor': "#000000",
            'steps': [{'range': [0, max_val], 'color': 'rgba(6, 182, 212, 0.15)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #E2E8F0; font-family: \'JetBrains Mono\', monospace; font-size: 15px; margin-bottom: 6px; border-bottom: 1px dashed rgba(217,70,239,0.3); padding-bottom:4px;">'
                        f'<span style="color: #d946ef; font-weight: bold; text-shadow: 1px 1px 0px #000;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.95); border: 4px solid #000000; padding: 25px; box-shadow: 8px 8px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 28px; color: #000000; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: #06b6d4; padding: 4px 12px; font-weight: 700;">SYNAPTIC UPLINK</div>
            {log_html if logs else '<div style="color: #64748b; font-family: \'JetBrains Mono\', monospace;">AWAITING NEURAL INPUT...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#d946ef"):
    st.markdown(f"""
        <div style="background: rgba(15, 17, 21, 0.9); border: 4px solid #000000; border-left: 15px solid {clr}; padding: 30px; margin-bottom: 35px; box-shadow: 10px 10px 0px #000000; position:relative; transform: skewX(1deg);">
            <div style="font-family:'Bebas Neue', cursive; font-size:34px; color:#FFFFFF; letter-spacing: 2px; margin-bottom:15px; text-shadow: 3px 3px 0px #000000;">
                {label}
            </div>
            <div style="font-size:16px; color:#0e0e0e; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 800; background: #FFFFFF; padding: 20px; border: 3px solid #000000; box-shadow: inset 3px 3px 0px rgba(0,0,0,0.15);">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
