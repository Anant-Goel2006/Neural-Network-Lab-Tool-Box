import streamlit as st

st.set_page_config(
    page_title="NEUROLAB",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from Perceptron.perceptron_ui                         import perceptron_page
from Forward_Propagation.forward_propagation          import forward_propagation_page
from Backward_Propagation.backward_propagation        import backward_propagation_page
from OpenCV_Detection.opencv_hub                      import opencv_detection_page
from Sentiment_Analysis.sentiment_analysis            import sentiment_analysis_page
from utils.styles                                     import inject_global_css

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
p_pct  = st.Page(perceptron_page,          title="1. The Perceptron",       icon="🟢")
p_fwd  = st.Page(forward_propagation_page, title="2. Forward Propagation",  icon="➡️")
p_bwd  = st.Page(backward_propagation_page,title="3. Backward Propagation", icon="⬅️")
p_cv   = st.Page(opencv_detection_page,    title="4. OpenCV Detection",     icon="📷")
p_sa   = st.Page(sentiment_analysis_page,  title="5. Sentiment Analysis",   icon="💬")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR BRANDING
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_brand():
    st.sidebar.markdown("""
    <div style="padding: 24px 20px; text-align:center; background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-bottom: 2px solid #3B82F6; border-radius: 12px; margin-bottom: 30px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: -40px; right: -40px; width: 100px; height: 100px; background: #3B82F6; opacity: 0.1; border-radius: 50%; filter: blur(25px);"></div>
        <div style="font-size: 56px; margin-bottom: 12px; filter: drop-shadow(0 4px 12px rgba(59, 130, 246, 0.3));">🧠</div>
        <div style="font-family:'Montserrat', sans-serif; font-weight: 800; font-size:30px; color:#F8FAFC; letter-spacing: 2px; line-height:1; text-transform: uppercase;">NEUROLAB</div>
        <div style="background: rgba(59, 130, 246, 0.1); color: #60A5FA; padding: 4px 12px; display: inline-block; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 11px; margin-top: 12px; text-transform: uppercase; letter-spacing: 1.5px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.2);">
            ULTIMATE EDITION
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home_page():
    inject_global_css()

    hero_img = r"C:\Users\konik\.gemini\antigravity\brain\08efec81-b5d1-4f14-94c7-3ba739dfee9a\neurolab_hero_banner_1774323520603.png"
    
    st.markdown(f"""
    <div class="hero-container fade-in" style="
        position: relative; 
        height: 500px; 
        border-radius: 24px; 
        overflow: hidden; 
        margin-bottom: 50px;
        background-image: linear-gradient(to bottom, rgba(15, 23, 42, 0.2), rgba(15, 23, 42, 1)), url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=2000&auto=format&fit=crop'); /* Fallback */
        background-size: cover;
        background-position: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 60px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
    ">
        <!-- Actual Background Image via Data URI or Path -->
        <style>
            .hero-container {{
                background-image: linear-gradient(to bottom, rgba(15, 23, 42, 0) 0%, rgba(15, 23, 42, 0.8) 70%, rgba(15, 23, 42, 1) 100%), url('{"file:///" + hero_img.replace('\\', '/')}');
            }}
            .see-modules-btn {{
                background: white !important;
                color: black !important;
                padding: 12px 30px !important;
                border-radius: 8px !important;
                font-weight: 700 !important;
                text-decoration: none !important;
                display: inline-flex !important;
                align-items: center !important;
                gap: 10px !important;
                transition: transform 0.2s, background 0.2s !important;
                font-family: 'Montserrat', sans-serif !important;
                font-size: 16px !important;
                width: fit-content;
            }}
            .see-modules-btn:hover {{
                transform: scale(1.05);
                background: #e5e5e5 !important;
            }}
        </style>

        <div style="position: relative; z-index: 2; max-width: 800px;">
            <h1 style="font-size: 80px; color: #F8FAFC; margin: 0; line-height: 0.9; font-weight: 900; font-family: 'Montserrat', sans-serif; letter-spacing: -2px; text-shadow: 0 0 40px rgba(0,0,0,0.5);">NEUROLAB</h1>
            <div style="margin: 20px 0; display: flex; align-items: center; gap: 15px;">
                <span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; padding: 4px 12px; font-weight: 700; font-size: 12px; border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.3); letter-spacing: 1px;">PREMIUM</span>
                <span style="color: #94A3B8; font-size: 14px; font-weight: 600;">2026 • ULTIMATE EDITION • 5 MODULES</span>
            </div>
            <p style="color: #E2E8F0; font-family: 'Inter', sans-serif; font-size: 18px; line-height: 1.5; margin-bottom: 30px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
                Experience the next generation of artificial intelligence. A cinematic playground for exploring the architectures that define our future.
            </p>
            <div style="display: flex; gap: 15px;">
                <a href="#modules-list" class="see-modules-btn">
                    <span style="font-size: 20px;">🔍</span> See Modules
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section Divider ─────────────────────────────────────────────────────
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:30px 0;">
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        <span style="font-size:14px;color:#94A3B8;letter-spacing:4px;
            font-weight:600;text-transform:uppercase; font-family:'Inter', sans-serif;">Lab Modules</span>
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
    </div>""", unsafe_allow_html=True)

    # ── Module Cards ────────────────────────────────────────────────────────
    # (Icon, Title, Subtitle, Description, Page, Tags, Color, ImagePath)
    CARDS = [
        ("🟢", "The Perceptron", "Binary Classifier",
         "Train a single neuron to find the optimal decision boundary in real-time.",
         p_pct, ["Live Training","6 Gates","Confusion Matrix","Weight Trajectory"], "#10B981", 
         r"C:\Users\konik\.gemini\antigravity\brain\4dcf7e85-c8ff-4bb1-803c-e8c5b7438e47\perceptron_module_banner_1774322293698.png"),

        ("➡️", "Forward Propagation", "Signal Flow Engine",
         "Trace neural signals through custom architectures with live activation heatmaps.",
         p_fwd, ["6 Activations","Layer Heatmap","Custom Arch","Weight Editor"], "#06B6D4",
         r"C:\Users\konik\.gemini\antigravity\brain\4dcf7e85-c8ff-4bb1-803c-e8c5b7438e47\forward_prop_module_banner_1774322313582.png"),

        ("⬅️", "Backward Propagation", "Gradient Engine",
         "Visualize the chain rule and gradient flow to understand neural learning.",
         p_bwd, ["Chain Rule Viz","Gradient Heatmap","Weight Diff","Live Gauges"], "#8B5CF6",
         r"C:\Users\konik\.gemini\antigravity\brain\4dcf7e85-c8ff-4bb1-803c-e8c5b7438e47\backward_prop_module_banner_1774322334899.png"),

        ("📷", "OpenCV Detection", "Vision Lab",
         "Advanced computer vision for face logging, vehicle counting, and gesture analytics.",
         p_cv, ["YOLO Vehicles","Live Analytics","Face Scanner","CSV Export"], "#F59E0B",
         r"C:\Users\konik\.gemini\antigravity\brain\4dcf7e85-c8ff-4bb1-803c-e8c5b7438e47\opencv_module_banner_1774322354099.png"),

        ("💬", "Sentiment Analysis", "LSTM Language Processor",
         "Deep language processing to detect 8 distinct emotions in real-time text.",
         p_sa, ["8 Emotions","LSTM Core","Mixed Sentiments","Batch Mode"], "#EC4899",
         r"C:\Users\konik\.gemini\antigravity\brain\4dcf7e85-c8ff-4bb1-803c-e8c5b7438e47\sentiment_analysis_banner_1774322370417.png"),
    ]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vertical List ──────────────────────────────────────────────────────
    st.markdown('<div id="modules-list"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-family: \'Montserrat\', sans-serif; margin-bottom: 25px; color: white; font-weight: 700; border-bottom: 2px solid #3B82F6; display: inline-block; padding-bottom: 10px;">Modules</h3>', unsafe_allow_html=True)
    
    for i, (ic, title, sub, desc, page, tags, clr, img) in enumerate(CARDS):
        with st.container():
            e_col1, e_col2, e_col3 = st.columns([1.2, 3, 1])
            
            with e_col1:
                st.image(img, use_container_width=True)
            
            with e_col2:
                st.markdown(f"""
                <div style="padding: 10px 0;">
                    <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 22px; color: white; margin-bottom: 5px;">{title}</div>
                    <div style="color: #3B82F6; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 15px; letter-spacing: 1px;">{sub}</div>
                    <p style="color: #F8FAFC; font-size: 15px; line-height: 1.6; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with e_col3:
                st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
                if st.button(f"Launch {title}", key=f"launch_v_{i}", type="primary", use_container_width=True):
                    st.switch_page(page)
            
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)

    # ── Tech Stack ──────────────────────────────────────────────────────────
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:40px 0 20px;">
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        <span style="font-size:14px;color:#94A3B8;letter-spacing:4px;
            font-weight:600;text-transform:uppercase; font-family:'Inter', sans-serif;">Core Tech Stack</span>
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
    </div>""", unsafe_allow_html=True)

    STACK = [("Streamlit","#FF4B4B"),("NumPy","#005BEA"),("Plotly","#ED1D24"),
             ("Pandas","#121212"),("OpenCV","#005BEA"),("TensorFlow","#FF6F00"),
             ("MediaPipe","#ED1D24"),("WebRTC","#005BEA")]
    st.markdown('<div style="text-align:center;line-height:2.8;">' +
        " ".join([f'<span style="background:#1E293B; border:1px solid #334155; color:#94A3B8;'
                  f'font-size:12px; padding:6px 14px; font-weight:600;'
                  f'letter-spacing:1px; margin:4px; text-transform:uppercase; border-radius:4px;">{n}</span>' for n,c in STACK]) +
        '</div>', unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:50px;padding-bottom:30px;">
        <div style="background: rgba(15, 23, 42, 0.4); display:inline-block; padding: 12px 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
            <span style="font-size:13px; color:#64748B; font-weight:500; font-family:'Inter', sans-serif; letter-spacing:1px;">
                NeuroLab Ultimate Suite © 2026 · Stable Release v5.0
            </span>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
p_home = st.Page(home_page, title="Dashboard", icon="🧠", default=True)

pages = {
    "Home":           [p_home],
    "Neural Network": [p_pct, p_fwd, p_bwd],
    "Applied AI":     [p_cv, p_sa],
}

inject_global_css()
sidebar_brand()
pg = st.navigation(pages)
pg.run()
