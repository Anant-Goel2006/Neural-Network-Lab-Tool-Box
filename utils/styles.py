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

        /* ──── NETFLIX-STYLE UI COMPONENTS ──── */
        .netflix-row-container {
            position: relative;
            padding: 20px 0;
            overflow: visible;
        }

        .netflix-row {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 20px 10px;
            scrollbar-width: none; /* Hide scrollbar Firefox */
            -ms-overflow-style: none; /* Hide scrollbar IE/Edge */
            scroll-behavior: smooth;
        }

        .netflix-row::-webkit-scrollbar {
            display: none; /* Hide scrollbar Chrome/Safari */
        }

        .netflix-card {
            flex: 0 0 280px;
            height: 160px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            position: relative;
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            cursor: pointer;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        .netflix-card:hover {
            transform: scale(1.15);
            z-index: 100;
            border-color: #3B82F6;
            box-shadow: 0 10px 40px rgba(0,0,0,0.8);
            background: rgba(30, 41, 59, 0.9);
        }

        .netflix-card-selected {
            border-bottom: 4px solid #3B82F6 !important;
            background: rgba(59, 130, 246, 0.1) !important;
        }

        .netflix-card-icon {
            font-size: 40px;
            margin-bottom: 5px;
            transition: all 0.3s ease;
        }

        .netflix-card-title {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 18px;
            color: #F8FAFC;
        }

        .netflix-detail-container {
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            margin-top: 20px;
            padding: 40px;
            animation: slideDown 0.5s ease-out;
            box-shadow: 0 20px 60px rgba(0,0,0,0.8);
            backdrop-filter: blur(25px);
            position: relative;
            z-index: 5;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .episode-card {
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px;
            display: flex;
            align-items: center;
            gap: 20px;
            transition: background 0.2s ease;
            cursor: pointer;
            border-radius: 4px;
        }

        .episode-card:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        .episode-number {
            font-size: 24px;
            font-weight: 600;
            color: #64748B;
            width: 40px;
        }

        .episode-info {
            flex: 1;
        }

        .episode-title {
            font-weight: 600;
            color: #F8FAFC;
            font-size: 16px;
        }

        .episode-desc {
            font-size: 13px;
            color: #94A3B8;
        }

        .play-button {
            background: #F8FAFC;
            color: #0F172A;
            padding: 10px 24px;
            border-radius: 4px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-right: 15px;
            transition: all 0.2s ease;
            text-decoration: none;
            cursor: pointer;
            border: none;
        }

        .play-button:hover {
            background: #E2E8F0;
            transform: scale(1.05);
        }

        .more-info-button {
            background: rgba(100, 116, 139, 0.4);
            color: #F8FAFC;
            padding: 10px 24px;
            border-radius: 4px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s ease;
            cursor: pointer;
            border: none;
        }

        .more-info-button:hover {
            background: rgba(100, 116, 139, 0.6);
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

                const THEMES = [
                    { name: "Cosmic", base: "#3B82F6", accent: "#60A5FA", pulse: "#8B5CF6" },
                    { name: "Marvel", base: "#EF4444", accent: "#F59E0B", pulse: "#FFFFFF" },
                    { name: "Batman", base: "#1E293B", accent: "#FACC15", pulse: "#475569" }
                ];

                class Neuron {
                    constructor() {
                        this.x = Math.random() * width;
                        this.y = Math.random() * height;
                        this.vx = (Math.random() - 0.5) * 0.4;
                        this.vy = (Math.random() - 0.5) * 0.4;
                        this.radius = Math.random() * 2 + 1.5;
                        this.firing = 0;
                        this.pulseSpeed = 0.012 + Math.random() * 0.02;
                        this.angle = Math.random() * Math.PI * 2;
                        this.spin = (Math.random() - 0.5) * 0.015;
                        this.theme = THEMES[Math.floor(Math.random() * THEMES.length)];
                    }
                    update() {
                        this.angle += this.spin;
                        this.x += this.vx + Math.cos(this.angle) * 0.15;
                        this.y += this.vy + Math.sin(this.angle) * 0.15;
                        
                        if(this.x < 0 || this.x > width) this.vx *= -1;
                        if(this.y < 0 || this.y > height) this.vy *= -1;
                        
                        if(Math.random() < 0.006 && this.firing === 0) {
                            this.firing = 1;
                        }
                        if(this.firing > 0) {
                            this.firing -= this.pulseSpeed;
                            if(this.firing < 0) this.firing = 0;
                        }
                    }
                    draw() {
                        // Dendrites with theme base color
                        ctx.beginPath();
                        ctx.strokeStyle = this.theme.base + (this.firing > 0.1 ? '44' : '1A'); // Hex + Alpha
                        ctx.lineWidth = 0.5;
                        for(let i=0; i<5; i++) {
                            const ang = (i / 5) * Math.PI * 2 + this.angle;
                            ctx.moveTo(this.x, this.y);
                            ctx.lineTo(this.x + Math.cos(ang) * 14, this.y + Math.sin(ang) * 14);
                        }
                        ctx.stroke();

                        // Nucleus with theme accent color
                        ctx.beginPath();
                        ctx.arc(this.x, this.y, this.radius + (this.firing * 3.5), 0, Math.PI * 2);
                        const bloom = this.firing * 18;
                        ctx.shadowBlur = 6 + bloom;
                        ctx.shadowColor = this.theme.accent;
                        ctx.fillStyle = this.firing > 0.5 ? this.theme.pulse : this.theme.accent;
                        ctx.fill();
                    }
                }

                const neurons = [];
                const neuronCount = 55; // Slightly reduced for multi-color complexity
                for(let i=0; i<neuronCount; i++) neurons.push(new Neuron());
                
                let signals = [];

                let lastTime = 0;
                function animate(time) {
                    if (time - lastTime < 33) {
                        parentDoc.defaultView.requestAnimationFrame(animate);
                        return;
                    }
                    lastTime = time;
                    ctx.clearRect(0, 0, width, height);
                    
                    const maxDistSq = 210 * 210;
                    
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
                                const midX = (n1.x + n2.x) / 2 + Math.cos(n1.angle) * 12;
                                const midY = (n1.y + n2.y) / 2 + Math.sin(n1.angle) * 12;
                                ctx.quadraticCurveTo(midX, midY, n2.x, n2.y);
                                
                                const alpha = (1 - (dist / 210)) * 0.1;
                                ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
                                ctx.lineWidth = 0.5;
                                ctx.shadowBlur = 0;
                                ctx.stroke();

                                if (n1.firing > 0.96 && Math.random() < 0.12) {
                                    signals.push({
                                        from: n1, to: n2, progress: 0, speed: 0.04 + Math.random() * 0.04, theme: n1.theme
                                    });
                                }
                            }
                        }
                    }

                    signals = signals.filter(s => {
                        s.progress += s.speed;
                        if (s.progress >= 1) {
                            s.to.firing = Math.min(1, s.to.firing + 0.25);
                            return false;
                        }
                        const t = s.progress;
                        const midX = (s.from.x + s.to.x) / 2 + Math.cos(s.from.angle) * 12;
                        const midY = (s.from.y + s.to.y) / 2 + Math.sin(s.from.angle) * 12;
                        
                        // Cubic-style curve for the signal pulse
                        const px = (1-t)*(1-t)*s.from.x + 2*(1-t)*t*midX + t*t*s.to.x;
                        const py = (1-t)*(1-t)*s.from.y + 2*(1-t)*t*midY + t*t*s.to.y;
                        
                        ctx.beginPath();
                        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
                        ctx.fillStyle = s.theme.accent;
                        ctx.shadowBlur = 12;
                        ctx.shadowColor = s.theme.accent;
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
