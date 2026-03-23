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

        /* ──── CLEAN GLASS CARDS (Native) ──── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.45) !important;
            backdrop-filter: blur(10px) saturate(160%) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
            position: relative;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6, #3B82F6);
            background-size: 200% 100%;
            animation: moveGradient 4s linear infinite;
            border-radius: 16px 16px 0 0;
            z-index: 10;
        }

        .premium-card {
            /* Keep for legacy manual HTML blocks in Hub/Dashboard */
            background: rgba(15, 23, 42, 0.45) !important;
            backdrop-filter: blur(12px) saturate(160%) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
            position: relative;
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
                        this.vx = (Math.random() - 0.5) * 0.5;
                        this.vy = (Math.random() - 0.5) * 0.5;
                        this.radius = Math.random() * 2 + 1.5;
                        this.firing = 0;
                        this.pulseSpeed = 0.015 + Math.random() * 0.02;
                        this.angle = Math.random() * Math.PI * 2;
                        this.spin = (Math.random() - 0.5) * 0.02;
                    }
                    update() {
                        // Organic drifting
                        this.angle += this.spin;
                        this.x += this.vx + Math.cos(this.angle) * 0.2;
                        this.y += this.vy + Math.sin(this.angle) * 0.2;
                        
                        if(this.x < 0 || this.x > width) this.vx *= -1;
                        if(this.y < 0 || this.y > height) this.vy *= -1;
                        
                        // Fire a pulse occasionally
                        if(Math.random() < 0.008 && this.firing === 0) {
                            this.firing = 1;
                        }
                        if(this.firing > 0) {
                            this.firing -= this.pulseSpeed;
                            if(this.firing < 0) this.firing = 0;
                        }
                    }
                    draw() {
                        // Draw Dendrites (small organic tendrils)
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(59, 130, 246, ${0.1 + this.firing * 0.2})`;
                        ctx.lineWidth = 0.5;
                        for(let i=0; i<6; i++) {
                            const ang = (i / 6) * Math.PI * 2 + this.angle;
                            ctx.moveTo(this.x, this.y);
                            ctx.lineTo(this.x + Math.cos(ang) * 12, this.y + Math.sin(ang) * 12);
                        }
                        ctx.stroke();

                        // Draw Nucleus
                        ctx.beginPath();
                        ctx.arc(this.x, this.y, this.radius + (this.firing * 4), 0, Math.PI * 2);
                        const bloom = this.firing * 20;
                        ctx.shadowBlur = 8 + bloom;
                        ctx.shadowColor = `rgba(96, 165, 250, ${0.4 + this.firing})`;
                        ctx.fillStyle = this.firing > 0 ? `rgba(255, 255, 255, ${0.9})` : 'rgba(96, 165, 250, 0.7)';
                        ctx.fill();
                    }
                }

                const neurons = [];
                const neuronCount = 60; // Optimized for organic feel
                for(let i=0; i<neuronCount; i++) neurons.push(new Neuron());
                
                // Synaptic signals (signals traveling between neurons)
                let signals = [];

                let lastTime = 0;
                function animate(time) {
                    if (time - lastTime < 33) {
                        parentDoc.defaultView.requestAnimationFrame(animate);
                        return;
                    }
                    lastTime = time;
                    ctx.clearRect(0, 0, width, height);
                    
                    const maxDistSq = 200 * 200;
                    
                    // Update and Draw Connections & Signals
                    for(let i=0; i<neuronCount; i++) {
                        for(let j=i+1; j<neuronCount; j++) {
                            const n1 = neurons[i];
                            const n2 = neurons[j];
                            const dx = n1.x - n2.x;
                            const dy = n1.y - n2.y;
                            const distSq = dx*dx + dy*dy;
                            
                            if(distSq < maxDistSq) {
                                const dist = Math.sqrt(distSq);
                                ctx.beginPath();
                                ctx.moveTo(n1.x, n1.y);
                                
                                // Organic curved axons
                                const midX = (n1.x + n2.x) / 2 + Math.cos(n1.angle) * 10;
                                const midY = (n1.y + n2.y) / 2 + Math.sin(n1.angle) * 10;
                                ctx.quadraticCurveTo(midX, midY, n2.x, n2.y);
                                
                                const alpha = (1 - (dist / 200)) * 0.12;
                                ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
                                ctx.lineWidth = 0.6;
                                ctx.shadowBlur = 0;
                                ctx.stroke();

                                // If n1 fires, create a synaptic signal traveling to n2
                                if (n1.firing > 0.95 && Math.random() < 0.1) {
                                    signals.push({
                                        from: n1, to: n2, progress: 0, speed: 0.05 + Math.random() * 0.05
                                    });
                                }
                            }
                        }
                    }

                    // Process and Draw Signals
                    signals = signals.filter(s => {
                        s.progress += s.speed;
                        if (s.progress >= 1) {
                            s.to.firing = Math.min(1, s.to.firing + 0.3); // Trigger receiver
                            return false;
                        }
                        const px = s.from.x + (s.to.x - s.from.x) * s.progress;
                        const py = s.from.y + (s.to.y - s.from.y) * s.progress;
                        
                        ctx.beginPath();
                        ctx.arc(px, py, 2, 0, Math.PI * 2);
                        ctx.fillStyle = "#60A5FA";
                        ctx.shadowBlur = 10;
                        ctx.shadowColor = "#3B82F6";
                        ctx.fill();
                        return true;
                    });
                    
                    neurons.forEach(n => {
                        n.update();
                        n.draw();
                    });
                    
                    parentDoc.defaultView.requestAnimationFrame(animate);
                }
                parentDoc.defaultView.requestAnimationFrame(animate);
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
