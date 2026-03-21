import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── MULTI-COLOR COMIC TEXT THEME & LAYOUT FIXES ──── */
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important; 
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── COMIC CARDS ──── */
        .premium-card {
            background: rgba(15, 23, 42, 0.85); /* Deep comic night blue/black */
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 3px solid #000000; 
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
            box-shadow: 9px 9px 0px #FACC15; /* Yellow jump */
            border-color: #000000;
        }

        h1, h2, h3 {
            font-family: 'Bebas Neue', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: 2px !important;
            text-transform: uppercase;
            word-wrap: break-word;
        }

        /* ──── COMIC BUTTONS ──── */
        .stButton > button {
            background: #EF4444 !important; /* Bold Red */
            border: 3px solid #000000 !important;
            border-radius: 2px !important;
            color: #ffffff !important;
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 20px !important;
            text-transform: uppercase !important;
            box-shadow: 5px 5px 0px #000000 !important;
            padding: 8px 20px !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 2px !important;
            white-space: normal !important; /* Prevents text overflow */
            height: auto !important;
        }
        .stButton > button:hover {
            background: #3B82F6 !important; /* Jump to Blue */
            color: #ffffff !important;
            transform: translate(-3px, -3px) !important;
            box-shadow: 8px 8px 0px #000000 !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 15, 25, 0.95) !important;
            backdrop-filter: blur(15px) !important;
            border-right: 3px solid #000000;
        }

        /* ──── METRICS ──── */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 38px !important; /* Reduced for overflow safety */
            color: #FACC15 !important;
            word-wrap: break-word;
            text-shadow: 2px 2px 0px #000000, -1px -1px 0px #000000, 1px -1px 0px #000000, -1px 1px 0px #000000; 
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT LIVE MULTI-COLOR COMIC NEURONS BACKGROUND ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('comic-live-bg')) {
                const canvas = parentDoc.createElement('canvas');
                canvas.id = 'comic-live-bg';
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
                
                // Vibrant Comic Source Colors
                const colors = ['#EF4444', '#3B82F6', '#FACC15', '#10B981', '#A855F7'];
                
                for(let i=0; i<particleCount; i++) {
                    particles.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        vx: (Math.random() - 0.5) * 2.0, /* Faster live movement */
                        vy: (Math.random() - 0.5) * 2.0,
                        radius: Math.random() * 3 + 1.5,
                        color: colors[Math.floor(Math.random() * colors.length)]
                    });
                }
                
                function animate() {
                    ctx.fillStyle = '#0f172a'; // Deep comic night 
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
                            
                            if(dist < 150) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                ctx.lineTo(p2.x, p2.y); // Snap straight lines for comic tech
                                
                                let alpha = (1 - (dist / 150)) * 0.4;
                                // Use source particle color for connection line
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
                const oldBio = parentDoc.getElementById('bio-cortex-bg');
                if(oldBio) oldBio.remove();
            }
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    # FIXED: Reduced font sizes and added word-wrap
    st.markdown(f"""
        <div style="margin-bottom:30px; position: relative; z-index: 10; transform: skewX(-2deg);">
            <div style="display:inline-block; border-bottom: 4px solid #3B82F6; padding-bottom: 5px; padding-right: 15px; box-shadow: 3px 3px 0px #000000; background: rgba(0,0,0,0.8); padding-left: 10px; border-left: 5px solid #3B82F6; max-width: 100%;">
                <h2 style="margin:0; font-size:28px; line-height: 1.1; text-shadow: 2px 2px 0px #000; word-wrap: break-word;">{title}</h2>
            </div>
            <p style="color:#e2e8f0; font-size:14px; font-weight:700; margin-top:10px; margin-left:10px; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing: 1px; text-shadow: 1px 1px 0px #000000; word-wrap: break-word;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    # FIXED: Reduced huge fonts causing layout breaking
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px); border: 4px solid #000000; padding:25px; 
            margin-bottom: 35px; box-shadow: 8px 8px 0px #EF4444; position:relative; overflow: hidden; transform: skewX(-1deg);">
            <div style="position: absolute; right: -10px; top: -10px; opacity: 0.1; font-size: 150px; font-family: 'Bebas Neue'; transform: rotate(15deg); color: #EF4444;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 20px; position: relative; z-index: 2; flex-wrap: wrap;">
                <div style="font-size: 55px; filter: drop-shadow(3px 3px 0px #000000);">{icon}</div>
                <div style="flex:1; min-width: 200px;">
                    <h1 style="font-size: 42px; margin: 0; line-height:1.1; color: #FFFFFF; text-shadow: 3px 3px 0px #000000, -1px -1px 0px #000, 1px -1px 0px #000, -1px 1px 0px #000; word-wrap: break-word;">{title}</h1>
                    <p style="color:#EF4444; font-size:18px; font-family:'Inter', sans-serif; font-weight: 800; letter-spacing:2px; margin-top:10px; text-transform: uppercase; text-shadow: 2px 2px 0px #000000; background: #000; display: inline-block; padding: 4px 10px; word-wrap: break-word;">
                        {sub}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def speedometer(val, max_val, title, color="#3B82F6", height=200):
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
            'steps': [{'range': [0, max_val], 'color': 'rgba(59, 130, 246, 0.15)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#FFFFFF", 'family': "Bebas Neue"}, height=height, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #E2E8F0; font-family: \'JetBrains Mono\', monospace; font-size: 14px; margin-bottom: 6px; border-bottom: 1px dashed rgba(59,130,246,0.3); padding-bottom:4px; word-wrap: break-word;">'
                        f'<span style="color: #3B82F6; font-weight: bold; text-shadow: 1px 1px 0px #000;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.95); border: 3px solid #000000; padding: 20px; box-shadow: 6px 6px 0px #000000; position: relative;">
            <div style="font-family: 'Bebas Neue', cursive; font-size: 24px; color: #FFFFFF; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 2px; display: inline-block; background: #3B82F6; padding: 2px 10px; font-weight: 700; border: 2px solid #000;">SYSTEM LOG</div>
            {log_html if logs else '<div style="color: #64748b; font-family: \'JetBrains Mono\', monospace;">SYS.STANDBY...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#10B981"):
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.9); border: 3px solid #000000; border-left: 10px solid {clr}; padding: 25px; margin-bottom: 25px; box-shadow: 8px 8px 0px #000000; position:relative; transform: skewX(1deg);">
            <div style="font-family:'Bebas Neue', cursive; font-size:26px; color:#FFFFFF; letter-spacing: 2px; margin-bottom:12px; text-shadow: 2px 2px 0px #000000; word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size:15px; color:#0f172a; font-family:'Inter', sans-serif; line-height: 1.5; font-weight: 800; background: #FFFFFF; padding: 15px; border: 2px solid #000000; box-shadow: inset 2px 2px 0px rgba(0,0,0,0.15); word-wrap: break-word;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
