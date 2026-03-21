import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── BIOLOGICAL CORTEX: SYNAPTIC GLOW THEME ──── */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── NEURAL ORGANIC CARDS ──── */
        .premium-card {
            background: rgba(17, 12, 34, 0.55); /* Deep neural violet */
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(139, 92, 246, 0.3); /* Synaptic purple border */
            border-top: 1px solid rgba(217, 70, 239, 0.5); /* Glowing magenta top line */
            border-radius: 20px; /* Smooth bio-organic edges */
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(139, 92, 246, 0.05); 
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
        }
        
        .premium-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(217, 70, 239, 0.15); /* Magenta glow float */
            border-color: rgba(217, 70, 239, 0.7);
            background: rgba(22, 15, 45, 0.75);
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            color: #ffffff !important;
            letter-spacing: 1px !important;
            font-weight: 700 !important;
        }

        /* ──── SYNAPTIC PULSE BUTTONS ──── */
        .stButton > button {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.8) 0%, rgba(217, 70, 239, 0.8) 100%) !important;
            border: 1px solid rgba(232, 121, 249, 0.5) !important;
            border-radius: 30px !important; /* Pill shaped bio-buttons */
            color: #ffffff !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            box-shadow: 0 4px 15px rgba(217, 70, 239, 0.3) !important;
            padding: 10px 28px !important;
            transition: all 0.3s ease !important;
            letter-spacing: 2px !important;
            backdrop-filter: blur(4px) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.9) 0%, rgba(56, 189, 248, 0.9) 100%) !important; /* Shifts to Cyan */
            border-color: rgba(34, 211, 238, 0.8) !important;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.6) !important;
            transform: translateY(-2px) scale(1.02);
            color: #ffffff !important;
        }

        /* ──── CORTEX SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(6, 4, 15, 0.85) !important;
            backdrop-filter: blur(25px) !important;
            border-right: 1px solid rgba(139, 92, 246, 0.2);
        }

        /* ──── METRICS (BIOLUMINESCENT) ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 46px !important;
            font-weight: 700 !important;
            color: #67e8f9 !important; /* Cyan bio-glow */
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.5) !important;
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
                        vx: (Math.random() - 0.5) * 0.8, /* Slow, floating organic movement */
                        vy: (Math.random() - 0.5) * 0.8,
                        radius: Math.random() * 3 + 0.5,
                        color: colors[Math.floor(Math.random() * colors.length)],
                        pulse: Math.random() * Math.PI * 2
                    });
                }
                
                function animate() {
                    // Deep brainstem background
                    ctx.fillStyle = '#06040f'; 
                    ctx.fillRect(0, 0, width, height);
                    
                    for(let i=0; i<particleCount; i++) {
                        let p = particles[i];
                        p.x += p.vx;
                        p.y += p.vy;
                        p.pulse += 0.05; // Firing synaptic pulse
                        
                        if(p.x < 0 || p.x > width) p.vx *= -1;
                        if(p.y < 0 || p.y > height) p.vy *= -1;
                        
                        // Pulsing radius to simulate firing neurons
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
                            
                            // Neuronal dendrite connections
                            if(dist < 140) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                
                                // Bezier curve for organic dendrite feel instead of rigid straight lines
                                let cx = (p.x + p2.x) / 2 + (Math.random() - 0.5) * 5;
                                let cy = (p.y + p2.y) / 2 + (Math.random() - 0.5) * 5;
                                
                                ctx.quadraticCurveTo(cx, cy, p2.x, p2.y);
                                
                                let alpha = (1 - (dist / 140)) * 0.6;
                                ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`; /* Violet connections */
                                ctx.lineWidth = 1;
                                ctx.shadowBlur = 0; // Performance optimization
                                ctx.stroke();
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
                
                // Cleanup of old background elements if they exist from previous renders
                const oldBat = parentDoc.getElementById('batman-live-bg');
                if(oldBat) oldBat.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:30px; position: relative; z-index: 10;">
            <div style="display:flex; align-items: center; gap: 15px; border-bottom: 1px solid rgba(217, 70, 239, 0.2); padding-bottom: 12px; margin-bottom: 12px;">
                <div style="width: 5px; height: 32px; background: #d946ef; border-radius: 10px; box-shadow: 0 0 15px rgba(217, 70, 239, 0.7);"></div>
                <h2 style="margin:0; font-size:32px; line-height: 1; filter: drop-shadow(0 0 12px rgba(217, 70, 239, 0.3));">{title}</h2>
            </div>
            <p style="color:#a78bfa; font-size:15px; font-weight:400; margin:0; font-family:'Inter', sans-serif; letter-spacing: 1px; opacity: 0.9;">
                {subtitle}
            </p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(17, 12, 34, 0.5); backdrop-filter: blur(20px); border: 1px solid rgba(139, 92, 246, 0.2); border-left: 4px solid #06b6d4; border-radius: 24px; padding:40px; 
            margin-bottom: 45px; box-shadow: 0 15px 40px rgba(0,0,0,0.5), inset 0 0 25px rgba(6, 182, 212, 0.05); position:relative; overflow: hidden;">
            <div style="position: absolute; right: 20px; top: 50%; transform: translateY(-50%); opacity: 0.05; font-size: 200px; font-family: 'Outfit'; filter: drop-shadow(0 0 40px #06b6d4);">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 30px; position: relative; z-index: 2;">
                <div style="font-size: 60px; color: #06b6d4; text-shadow: 0 0 30px rgba(6, 182, 212, 0.5);">{icon}</div>
                <div style="flex:1;">
                    <h1 style="font-size: 48px; margin: 0; line-height:1.2; color: #ffffff; letter-spacing: 1px;">{title}</h1>
                    <p style="color:#d946ef; font-size:16px; font-family:'Outfit', sans-serif; font-weight: 500; letter-spacing:3px; margin-top:10px; text-transform: uppercase;">
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
        title = {'text': title.upper(), 'font': {'size': 18, 'color': '#ffffff', 'family': 'Outfit'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': color, 'thickness': 0.65},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 0,
            'steps': [{'range': [0, max_val], 'color': 'rgba(6, 182, 212, 0.08)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff", 'family': "Outfit"}, height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #cbd5e1; font-family: \'JetBrains Mono\', monospace; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:6px;">'
                        f'<span style="color: #d946ef; font-weight: 600; text-shadow: 0 0 8px rgba(217,70,239,0.5);">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(12, 9, 26, 0.7); backdrop-filter: blur(15px); border: 1px solid rgba(139, 92, 246, 0.15); padding: 24px; border-radius: 20px; box-shadow: inset 0 0 25px rgba(0,0,0,0.8); border-top: 2px solid #06b6d4;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 16px; color: #06b6d4; margin-bottom: 18px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; display: flex; align-items: center; gap: 12px;">
                <div style="width: 10px; height: 10px; background: #06b6d4; border-radius: 50%; box-shadow: 0 0 12px #06b6d4, 0 0 24px #06b6d4;"></div>
                SYNAPTIC FEED // LIVE
            </div>
            {log_html if logs else '<div style="color: #64748b; font-family: \'JetBrains Mono\', monospace;">AWAITING NEURAL INPUT...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#d946ef"):
    st.markdown(f"""
        <div style="background: rgba(17, 12, 34, 0.55); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); border: 1px solid rgba(139, 92, 246, 0.2); border-left: 4px solid {clr}; padding: 26px; margin-bottom: 30px; box-shadow: 0 10px 35px rgba(0,0,0,0.5); position:relative; border-radius: 20px;">
            <div style="font-family:'Outfit', sans-serif; font-size:20px; color:#ffffff; letter-spacing: 2px; margin-bottom:16px; text-transform: uppercase; font-weight: 700; text-shadow: 0 0 15px {clr}; display: flex; align-items: center; gap: 12px;">
                <span style="color: {clr}; opacity: 0.9;">●</span> {label}
            </div>
            <div style="font-size:14px; color:#f8fafc; font-family:'Inter', sans-serif; line-height: 1.7; font-weight: 400; background: rgba(0,0,0,0.3); padding: 20px; border: 1px solid rgba(255,255,255,0.03); border-radius: 12px; box-shadow: inset 0 0 15px rgba(0,0,0,0.4);">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
