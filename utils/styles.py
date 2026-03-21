import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── BATMAN COMIC EDITION: LIVE NEURAL BACKGROUND ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; /* Let the animated canvas show through */
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
        }

        /* ──── TACTICAL BAT-ARMOR CARDS ──── */
        .premium-card {
            background: rgba(15, 17, 21, 0.85); /* Dark tactical Kevlar */
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 3px solid #FACC15; /* Striking Bat-Yellow border */
            border-radius: 0px; /* Sharp tactical comic edges */
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 8px 8px 0px #000000; /* Solid Comic black shadow */
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
        }
        
        .premium-card:hover {
            transform: translate(-4px, -4px);
            box-shadow: 12px 12px 0px #FACC15; /* Yellow jump on hover */
            border-color: #FFFFFF;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase;
        }

        /* ──── TACTICAL BAT-BUTTONS ──── */
        .stButton > button {
            background: #000000 !important; /* Pitch Black */
            border: 3px solid #FACC15 !important;
            border-radius: 0px !important;
            color: #FACC15 !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 24px !important;
            text-transform: uppercase !important;
            box-shadow: 5px 5px 0px #FACC15 !important;
            padding: 10px 24px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
        }
        .stButton > button:hover {
            background: #FACC15 !important; 
            color: #000000 !important;
            border-color: #000000 !important;
            transform: translate(-3px, -3px) !important;
            box-shadow: 8px 8px 0px #000000 !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(5, 7, 10, 0.95) !important;
            border-right: 3px solid #FACC15;
            backdrop-filter: blur(10px) !important;
        }

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 46px !important;
            color: #FACC15 !important;
            text-shadow: 3px 3px 0px #000000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000; 
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT LIVE ANIMATED HTML5 CANVAS FOR JS NEURON BACKGROUND ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('batman-live-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'batman-live-bg';
                canvas.style.position = 'fixed';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100vw';
                canvas.style.height = '100vh';
                canvas.style.zIndex = '-1'; /* Keep it firmly in the background */
                canvas.style.pointerEvents = 'none';
                
                // Insert it immediately into the body
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
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 1.5,
                        vy: (Math.random() - 0.5) * 1.5,
                        radius: Math.random() * 2.5 + 1
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#06080d'; // Deep Gotham Black
                    ctx.fillRect(0, 0, width, height);
                    
                    for(let i=0; i<particleCount; i++) {
                        let p = particles[i];
                        p.x += p.vx;
                        p.y += p.vy;
                        
                        if(p.x < 0 || p.x > width) p.vx *= -1;
                        if(p.y < 0 || p.y > height) p.vy *= -1;
                        
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                        // Make mostly gold/yellow Bat-Signal nodes, occasionally dark/steel grey
                        ctx.fillStyle = i % 3 === 0 ? '#475569' : '#facc15'; 
                        ctx.fill();
                        
                        for(let j=i+1; j<particleCount; j++) {
                            let p2 = particles[j];
                            let dx = p.x - p2.x;
                            let dy = p.y - p2.y;
                            let dist = Math.sqrt(dx*dx + dy*dy);
                            
                            // Max distance for neural transmission connections
                            if(dist < 130) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                ctx.lineTo(p2.x, p2.y);
                                
                                // Lines fade as distance increases
                                let alpha = 1 - (dist / 130);
                                ctx.strokeStyle = `rgba(250, 204, 21, ${alpha * 0.4})`;
                                ctx.lineWidth = 1;
                                ctx.stroke();
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:35px; position: relative; z-index: 10; transform: skewX(-4deg);">
            <div style="display:inline-block; border-bottom: 5px solid #FACC15; padding-bottom: 5px; padding-right: 20px; box-shadow: 4px 4px 0px #000000; background: rgba(0,0,0,0.8); padding-left: 15px; border-left: 5px solid #FACC15;">
                <h2 style="margin:0; font-size:40px; line-height: 1; text-shadow: 2px 2px 0px #000;">{title}</h2>
            </div>
            <p style="color:#D1D5DB; font-size:16px; font-weight:700; margin-top:12px; margin-left:15px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 2px; text-shadow: 1px 1px 0px #000000;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.9); backdrop-filter: blur(10px); border: 5px solid #FACC15; padding:40px; 
            margin-bottom: 45px; box-shadow: 12px 12px 0px #000000; position:relative; overflow: hidden; transform: skewX(-2deg);">
            <div style="position: absolute; right: -20px; top: -50px; opacity: 0.1; font-size: 250px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #FACC15;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 30px; position: relative; z-index: 2;">
                <div style="font-size: 75px; filter: drop-shadow(4px 4px 0px #000000);">{icon}</div>
                <div style="flex:1;">
                    <h1 style="font-size: 68px; margin: 0; line-height:1; color: #FFFFFF; text-shadow: 4px 4px 0px #000000, -2px -2px 0px #000, 2px -2px 0px #000, -2px 2px 0px #000;">{title}</h1>
                    <p style="color:#FACC15; font-size:24px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:3px; margin-top:15px; text-transform: uppercase; text-shadow: 2px 2px 0px #000000; background: #000; display: inline-block; padding: 4px 12px;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#FACC15", height=220):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 26, 'color': '#FFFFFF', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 4, 'tickcolor': "#FACC15"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.8)",
            'borderwidth': 4,
            'bordercolor': "#FACC15",
            'steps': [{'range': [0, max_val], 'color': 'rgba(250, 204, 21, 0.15)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #E2E8F0; font-family: \'JetBrains Mono\', monospace; font-size: 15px; margin-bottom: 6px; border-bottom: 1px dashed rgba(250,204,21,0.3); padding-bottom:4px;">'
                        f'<span style="color: #FACC15; font-weight: bold; text-shadow: 1px 1px 0px #000;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(10, 12, 16, 0.95); border: 4px solid #FACC15; padding: 25px; box-shadow: 8px 8px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 28px; color: #000000; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: #FACC15; padding: 4px 12px; font-weight: 700;">BATCOMPUTER UPLINK</div>
            {log_html if logs else '<div style="color: #94A3B8; font-family: \'JetBrains Mono\', monospace;">SYS.STANDBY...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#FACC15"):
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
