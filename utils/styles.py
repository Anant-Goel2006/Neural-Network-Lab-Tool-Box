import streamlit as st
import time
import plotly.graph_objects as go
import streamlit.components.v1 as components

def inject_global_css():
    st.markdown("""
        <style>
        /* ──── PREMIUM DEEP LEARNING THEME ──── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Montserrat:wght@500;600;700;800&family=JetBrains+Mono&display=swap');

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], .main, [data-testid="stBlockContainer"], [data-testid="stVerticalBlock"], .block-container, section {
            background-color: transparent !important; 
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ──── ULTRA-PREMIUM GLASS CARDS ──── */
        .premium-card {
            background: rgba(15, 23, 42, 0.4); 
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08); 
            border-radius: 16px; 
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5); 
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            z-index: 2;
            word-wrap: break-word; 
            overflow: hidden;
        }
        
        .premium-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6, #3B82F6);
            background-size: 200% 100%;
            animation: moveGradient 4s linear infinite;
        }

        @keyframes moveGradient {
            0% { background-position: 100% 0%; }
            100% { background-position: -100% 0%; }
        }

        .premium-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
            border-color: rgba(59, 130, 246, 0.4);
            background: rgba(15, 23, 42, 0.6);
        }

        h1, h2, h3 {
            font-family: 'Montserrat', sans-serif !important;
            color: #F8FAFC !important;
            letter-spacing: 0.5px !important;
            font-weight: 700 !important;
        }

        /* ──── GLOWING BUTTONS ──── */
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
            padding: 12px 28px !important;
            transition: all 0.3s ease !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6) !important;
            background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
        }

        /* ──── SIDEBAR ──── */
        [data-testid="stSidebar"] {
            background-color: rgba(7, 10, 15, 0.95) !important;
            backdrop-filter: blur(25px) !important;
            border-right: 1px solid rgba(59, 130, 246, 0.2);
        }

        /* ──── CUSTOM SCROLLBAR ──── */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0F172A;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ── INJECT ELECTRIFYING NEURONS LIVE CANVAS ──
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            const oldCanvas = parentDoc.getElementById('neuro-bg-canvas');
            if (oldCanvas) oldCanvas.remove();
            
            const canvas = parentDoc.createElement('canvas');
            canvas.id = 'neuro-bg-canvas';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.zIndex = '-1'; 
            canvas.style.pointerEvents = 'none';
            canvas.style.background = 'radial-gradient(circle at center, #0B0E14 0%, #020205 100%)';
            
            parentDoc.body.insertBefore(canvas, parentDoc.body.firstChild);
                
                const ctx = canvas.getContext('2d');
                let width = canvas.width = parentDoc.defaultView.innerWidth;
                let height = canvas.height = parentDoc.defaultView.innerHeight;
                
                parentDoc.defaultView.addEventListener('resize', () => {
                    width = canvas.width = parentDoc.defaultView.innerWidth;
                    height = canvas.height = parentDoc.defaultView.innerHeight;
                });

                class Neuron {
                    constructor() {
                        this.x = Math.random() * width;
                        this.y = Math.random() * height;
                        this.vx = (Math.random() - 0.5) * 0.8;
                        this.vy = (Math.random() - 0.5) * 0.8;
                        this.radius = Math.random() * 2.5 + 1;
                        this.firing = 0; // 0 to 1
                        this.pulseSpeed = 0.02 + Math.random() * 0.03;
                    }
                    update() {
                        this.x += this.vx;
                        this.y += this.vy;
                        if(this.x < 0 || this.x > width) this.vx *= -1;
                        if(this.y < 0 || this.y > height) this.vy *= -1;
                        
                        if(Math.random() < 0.01 && this.firing === 0) {
                            this.firing = 1;
                        }
                        if(this.firing > 0) {
                            this.firing -= this.pulseSpeed;
                            if(this.firing < 0) this.firing = 0;
                        }
                    }
                    draw() {
                        ctx.beginPath();
                        ctx.arc(this.x, this.y, this.radius + (this.firing * 3), 0, Math.PI * 2);
                        const bloom = this.firing * 15;
                        ctx.shadowBlur = 5 + bloom;
                        ctx.shadowColor = this.firing > 0 ? `rgba(96, 165, 250, ${0.5 + this.firing})` : 'rgba(59, 130, 246, 0.3)';
                        ctx.fillStyle = this.firing > 0 ? `rgba(240, 249, 255, ${0.8 + this.firing * 0.2})` : 'rgba(59, 130, 246, 0.6)';
                        ctx.fill();
                    }
                }

                const neurons = [];
                const neuronCount = 85; 
                for(let i=0; i<neuronCount; i++) neurons.push(new Neuron());
                
                function animate() {
                    ctx.clearRect(0, 0, width, height);
                    
                    // Draw connections first
                    for(let i=0; i<neuronCount; i++) {
                        for(let j=i+1; j<neuronCount; j++) {
                            const n1 = neurons[i];
                            const n2 = neurons[j];
                            const dx = n1.x - n2.x;
                            const dy = n1.y - n2.y;
                            const dist = Math.sqrt(dx*dx + dy*dy);
                            
                            if(dist < 180) {
                                ctx.beginPath();
                                ctx.moveTo(n1.x, n1.y);
                                ctx.lineTo(n2.x, n2.y);
                                
                                const alpha = (1 - (dist / 180)) * 0.15;
                                const isFiring = (n1.firing > 0.5 || n2.firing > 0.5);
                                
                                if(isFiring) {
                                    ctx.strokeStyle = `rgba(96, 165, 250, ${alpha * 4})`;
                                    ctx.lineWidth = 1.8;
                                    ctx.shadowBlur = 10;
                                    ctx.shadowColor = '#60A5FA';
                                } else {
                                    ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
                                    ctx.lineWidth = 0.8;
                                    ctx.shadowBlur = 0;
                                }
                                ctx.stroke();
                            }
                        }
                    }
                    
                    neurons.forEach(n => {
                        n.update();
                        n.draw();
                    });
                    
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                animate();
        </script>
    """, height=0, width=0)

def section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom:24px; position: relative; z-index: 10;">
            <div style="display:inline-block; border-bottom: 2px solid #3B82F6; padding-bottom: 8px; max-width: 100%;">
                <h2 style="margin:0; font-size:28px; line-height: 1.2; color: #F8FAFC; word-wrap: break-word; font-weight: 700;">{title}</h2>
            </div>
            <p style="color:#94a3b8; font-size:14px; font-weight:500; margin-top:10px; font-family:'Inter', sans-serif; letter-spacing: 0.5px; word-wrap: break-word;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def gradient_header(title, sub, icon=""):
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.05); border-left: 4px solid #3B82F6; padding:32px; 
            margin-bottom: 35px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); position:relative; overflow: hidden; border-radius: 12px;">
            <div style="position: absolute; right: 20px; top: 50%; transform: translateY(-50%); opacity: 0.05; font-size: 120px; color: #3B82F6;">
                {icon}
            </div>
            <div style="display: flex; align-items: center; gap: 24px; position: relative; z-index: 2; flex-wrap: wrap;">
                <div style="font-size: 48px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));">{icon}</div>
                <div style="flex:1; min-width: 200px;">
                    <h1 style="font-size: 34px; margin: 0; line-height:1.2; color: #FFFFFF; font-weight: 700; word-wrap: break-word;">{title}</h1>
                    <p style="color:#DBEAFE; font-size:15px; font-family:'Inter', sans-serif; font-weight: 500; margin-top:8px; display: inline-block; background: rgba(59, 130, 246, 0.15); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.25); word-wrap: break-word;">
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
        title = {'text': title.upper(), 'font': {'size': 14, 'color': '#94A3B8', 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(15,23,42,0.5)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [{'range': [0, max_val], 'color': 'rgba(59, 130, 246, 0.03)'}]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#F8FAFC", 'family': "Montserrat"}, height=height, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def render_log(placeholder, logs):
    log_html = "".join([f'<div style="color: #94A3B8; font-family: \'JetBrains Mono\', monospace; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:6px; word-wrap: break-word;">'
                        f'<span style="color: #3B82F6; font-weight: 500;">[{time.strftime("%H:%M:%S")}]</span> {msg}</div>' for msg in logs[-8:]])
    placeholder.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-top: 3px solid #3B82F6; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); position: relative; border-radius: 8px;">
            <div style="font-family: 'Montserrat', sans-serif; font-size: 16px; color: #F8FAFC; margin-bottom: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></div>
                SYSTEM STATUS LOG
            </div>
            {log_html if logs else '<div style="color: #64748B; font-family: \'JetBrains Mono\', monospace; font-size: 13px;">AWAITING SYSTEM BOOT...</div>'}
        </div>
    """, unsafe_allow_html=True)

def render_nlp_insight(text, label, clr="#3B82F6"):
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-left: 4px solid {clr}; padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); position:relative; border-radius: 8px;">
            <div style="font-family:'Montserrat', sans-serif; font-size:18px; color:#F8FAFC; font-weight: 600; margin-bottom:10px; word-wrap: break-word;">
                {label}
            </div>
            <div style="font-size:14px; color:#CBD5E1; font-family:'Inter', sans-serif; line-height: 1.6; font-weight: 400; background: rgba(0,0,0,0.15); padding: 16px; border: 1px solid rgba(255,255,255,0.05); word-wrap: break-word; border-radius: 6px;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
