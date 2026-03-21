import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── METALLIC BLACK & SPIDER-MAN SUIT THEME ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── METALLIC SPIDER-SUIT CARDS ──── */
        .premium-card {
            background: linear-gradient(145deg, #16181a, #08090a); /* Metallic brushed Kevlar */
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 2px solid #1f2228; /* Gunmetal structure */
            border-top: 3px solid #DC2626; /* Web-Slinger Red */
            border-bottom: 3px solid #3B82F6; /* Suit Blue */
            border-radius: 6px; 
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.9), inset 0 0 15px rgba(220, 38, 38, 0.05); 
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            z-index: 2;
            word-wrap: break-word; 
        }
        
        .premium-card:hover {
            transform: translateY(-4px);
            box-shadow: 12px 12px 25px rgba(0, 0, 0, 0.9), inset 0 0 20px rgba(220, 38, 38, 0.15);
            border-color: #DC2626;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2.5px !important;
            text-transform: uppercase;
            word-wrap: break-word;
        }

        /* ──── SPIDER ACTION BUTTONS ──── */
        .stButton > button {
            background: linear-gradient(180deg, #DC2626 0%, #991B1B 100%) !important; /* Spider Red Metallic */
            border: 2px solid #EF4444 !important;
            border-radius: 4px !important;
            color: #ffffff !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 22px !important;
            text-transform: uppercase !important;
            box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4) !important;
            padding: 10px 24px !important;
            transition: all 0.2s ease-in-out !important;
            letter-spacing: 2px !important;
            white-space: normal !important; 
            height: auto !important;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, #3B82F6 0%, #1D4ED8 100%) !important; /* Spider Blue Metallic */
            border-color: #60A5FA !important;
            color: #ffffff !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(8, 9, 10, 0.95) !important;
            backdrop-filter: blur(15px) !important;
            border-right: 2px solid #DC2626;
        }

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 40px !important; 
            color: #DC2626 !important; /* Spider Red */
            word-wrap: break-word;
            text-shadow: 2px 2px 0px #000000, 0 0 15px rgba(220, 38, 38, 0.5); 
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT SPIDER-WEB METALLIC LIVE CANVAS ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('spider-metallic-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'spider-metallic-bg';
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
                
                // Metallic Spider Suit Palette
                const colors = ['#DC2626', '#B91C1C', '#3B82F6', '#ffffff', '#64748b'];
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 2.5, // High agility spider movement
                        vy: (Math.random() - 0.5) * 2.5,
                        radius: Math.random() * 2.5 + 1,
                        color: colors[Math.floor(Math.random() * colors.length)]
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#050505'; // Metallic pitch black 
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
                        ctx.shadowBlur = 10;
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
                                ctx.lineTo(p2.x, p2.y); 
                                
                                let alpha = (1 - (dist / 140)) * 0.5;
                                ctx.globalAlpha = alpha;
                                ctx.strokeStyle = '#e2e8f0'; // Web-Slinger Silver/White connections
                                ctx.lineWidth = 1.2;
                                ctx.shadowBlur = 0; // Performance
                                ctx.stroke();
                                ctx.globalAlpha = 1.0;
                            }
                        }
                    }
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
                
                // Cleanup old light background
                const oldLight = parentDoc.getElementById('light-comic-bg');
                if(oldLight) oldLight.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:30px; position: relative; z-index: 10; transform: skewX(-2deg);">
            <div style="display:inline-block; border-bottom: 2px solid #3B82F6; padding-bottom: 5px; padding-right: 20px; box-shadow: 0 5px 15px rgba(220,38,38,0.2); background: linear-gradient(90deg, #111111, transparent); padding-left: 15px; border-left: 5px solid #DC2626; max-width: 100%;">
                <h2 style="margin:0; font-size:32px; line-height: 1.1; color: #FFFFFF; text-shadow: 2px 2px 0px #000; word-wrap: break-word;">{title}</h2>
            </div>
            <p style="color:#94a3b8; font-size:15px; font-weight:700; margin-top:12px; margin-left:15px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 1.5px; word-wrap: break-word;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #151515 0%, #050505 100%); border: 2px solid #222222; border-left: 5px solid #DC2626; padding:30px; 
            margin-bottom: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.9); position:relative; overflow: hidden; transform: skewX(-1deg); border-radius: 6px;">
            <div style="position: absolute; right: -10px; top: -10px; opacity: 0.05; font-size: 150px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #DC2626;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 2; flex-wrap: wrap;">
                <div style="font-size: 60px; filter: drop-shadow(0 0 15px rgba(220,38,38,0.5));">{icon}</div>
                <div style="flex:1; min-width: 200px;">
                    <h1 style="font-size: 46px; margin: 0; line-height:1.1; color: #FFFFFF; text-shadow: 2px 2px 0px #000000, 0 0 10px rgba(255,255,255,0.2); word-wrap: break-word;">{title}</h1>
                    <p style="color:#FFFFFF; font-size:18px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:2px; margin-top:12px; text-transform: uppercase; text-shadow: none; background: #DC2626; display: inline-block; padding: 4px 12px; border-radius: 2px; box-shadow: 0 4px 10px rgba(220,38,38,0.4); word-wrap: break-word;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#DC2626", height=200):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title.upper(), 'font': {'size': 20, 'color': '#FFFFFF', 'family': 'Bebas Neue'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [{'range': [0, max_val], 'color': 'rgba(220, 38, 38, 0.1)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #cbd5e1; font-family: \'JetBrains Mono\', monospace; font-size: 14px; margin-bottom: 6px; border-bottom: 1px dashed #334155; padding-bottom:4px; word-wrap: break-word;">'
                        f'<span style="color: #DC2626; font-weight: bold; text-shadow: 0 0 5px rgba(220,38,38,0.5);">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: linear-gradient(145deg, #111111, #050505); border: 2px solid #1f2228; border-top: 3px solid #DC2626; padding: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.9); position: relative; border-radius: 4px;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 24px; color: #FFFFFF; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: transparent; padding: 0px; font-weight: 700; text-shadow: 0 0 10px #DC2626;">SPIDER-SUIT UPLINK</div>
            {log_html if logs else '<div style="color: #475569; font-family: \'JetBrains Mono\', monospace;">AWAITING SYSTEM BOOT...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#3B82F6"):
    st.markdown(f"""
        <div style="background: linear-gradient(145deg, #111111, #050505); border: 2px solid #1f2228; border-left: 6px solid {clr}; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.9); position:relative; border-radius: 4px;">
            <div style="font-family:'Bebas Neue', cursive; font-size:28px; color:#FFFFFF; letter-spacing: 2px; margin-bottom:12px; text-shadow: 0 0 15px rgba(59,130,246,0.6); word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size:15px; color:#e2e8f0; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 400; background: rgba(0,0,0,0.5); padding: 15px; border: 1px solid #334155; box-shadow: inset 0 2px 10px rgba(0,0,0,0.5); word-wrap: break-word; border-radius: 2px;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
