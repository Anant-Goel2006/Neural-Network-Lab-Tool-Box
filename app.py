import streamlit as st
import streamlit.components.v1 as components
import os
from utils.styles import sidebar_brand, inject_global_css, get_image_base64

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEUROLAB",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SHARED UI ──
inject_global_css()
sidebar_brand()

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home_page():
    inject_global_css()

    hero_img_path = os.path.join("assets", "banners", "neurolab_netflix_style_hero_1774323980383.png")
    hero_base64 = get_image_base64(hero_img_path)
    
    st.markdown(f"""
    <style>
        .hero-container-main {{
            position: relative; 
            min-height: 450px; 
            width: 100%;
            border-radius: 16px; 
            overflow: hidden; 
            margin-bottom: 40px;
            background: linear-gradient(to right, rgba(15, 23, 42, 1) 0%, rgba(15, 23, 42, 0.4) 60%, rgba(15, 23, 42, 0) 100%), url('{hero_base64}');
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            padding: 0 60px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        }}
        .hero-title-text {{
            font-size: 68px !important; 
            color: #F8FAFC !important; 
            margin: 0 !important; 
            line-height: 1 !important; 
            font-weight: 900 !important; 
            font-family: 'Montserrat', sans-serif !important; 
            letter-spacing: -2px !important;
            white-space: nowrap;
            text-shadow: 0 4px 20px rgba(0,0,0,0.8);
        }}
        .hero-btn-play {{
            background: white !important; 
            color: black !important; 
            padding: 12px 30px !important; 
            border-radius: 4px !important; 
            font-weight: 700 !important; 
            border: none !important; 
            font-size: 16px !important; 
            cursor: pointer !important; 
            display: inline-flex !important; 
            align-items: center !important; 
            gap: 10px !important; 
            transition: 0.2s !important; 
            font-family: 'Montserrat', sans-serif !important;
            text-decoration: none !important;
        }}
        .hero-btn-play:hover {{ transform: scale(1.05); background: #eee !important; }}
    </style>
    
    <div class="hero-container-main">
        <div style="position: relative; z-index: 2; max-width: 650px; width: 100%;">
            <h1 class="hero-title-text">NEUROLAB</h1>
            <div style="margin: 20px 0; display: flex; align-items: center; gap: 12px;">
                <span style="background: #E50914; color: white; padding: 4px 10px; font-weight: 800; font-size: 12px; border-radius: 2px; letter-spacing: 0.5px;">ULTIMATE</span>
                <span style="color: #94A3B8; font-size: 14px; font-weight: 600;">2026 • 7 MODULES</span>
            </div>
            <p style="color: #E2E8F0; font-family: 'Inter', sans-serif; font-size: 18px; line-height: 1.5; margin-bottom: 30px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
                Experience the next generation of artificial intelligence. A cinematic playground for exploring the architectures that define our future.
            </p>
            <div style="display: flex; gap: 15px;">
                <a href="#lab-modules" class="hero-btn-play">
                    <span style="font-size: 20px;">▶</span> See Modules
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    components.html("""
        <script>
            setTimeout(() => {
                const links = window.parent.document.querySelectorAll('a[href="#lab-modules"]');
                links.forEach(link => {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        const el = window.parent.document.getElementById('modules-list');
                        if(el) el.scrollIntoView({behavior: 'smooth'});
                    });
                });
            }, 1000);
        </script>
    """, height=0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:30px 0;">
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        <span style="font-size:14px;color:#94A3B8;letter-spacing:4px;
            font-weight:600;text-transform:uppercase; font-family:'Inter', sans-serif;">Lab Modules</span>
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
    </div>""", unsafe_allow_html=True)

    CARDS = [
        ("🟢", "The Perceptron", "Binary Classifier",
         "Train a single neuron to find the optimal decision boundary in real-time.",
         p_pct, ["Live Training","6 Gates","Confusion Matrix","Weight Trajectory"], "#10B981", 
         os.path.join("assets", "banners", "perceptron_module_banner_1774322293698.png")),

        ("➡️", "Forward Propagation", "Signal Flow Engine",
         "Trace neural signals through custom architectures with live activation heatmaps.",
         p_fwd, ["6 Activations","Layer Heatmap","Custom Arch","Weight Editor"], "#06B6D4",
         os.path.join("assets", "banners", "forward_prop_module_banner_1774322313582.png")),

        ("⬅️", "Backward Propagation", "Gradient Engine",
         "Visualize the chain rule and gradient flow to understand neural learning.",
         p_bwd, ["Chain Rule Viz","Gradient Heatmap","Weight Diff","Live Gauges"], "#8B5CF6",
         os.path.join("assets", "banners", "backward_prop_module_banner_1774322334899.png")),

        ("🧠", "Hopfield Network", "Associative Memory",
         "Explore content-addressable memory and energy minimization in a recurrent neural architecture.",
         p_hop, ["Associative Memory","Energy Landscape","Pattern Recovery","Hebbian Learning"], "#3B82F6",
         os.path.join("assets", "banners", "hopfield_module_banner_1775024868657.png")),

        ("📷", "OpenCV Detection", "Vision Lab",
         "Advanced computer vision for face logging, vehicle counting, and gesture analytics.",
         p_cv, ["YOLO Vehicles","Live Analytics","Face Scanner","CSV Export"], "#F59E0B",
         os.path.join("assets", "banners", "opencv_module_banner_1774322354099.png")),

        ("💬", "Sentiment Analysis", "Deep Sentiment Engine",
         "Deep language processing to detect 8 distinct emotions in real-time text.",
         p_sa, ["8 Emotions","LSTM Core","Mixed Sentiments","Batch Mode"], "#EC4899",
         os.path.join("assets", "banners", "sentiment_analysis_banner_1774322370417.png")),

        ("🚀", "LSTM Hub", "Next-Gen Sequence Lab",
         "A refined powerhouse of LSTM modules including Dynamic Word Prediction, Sentiment HUD, and Creative Story Engines.",
         p_hub, ["Word Prediction","Sentiment HUD","Creative Gen","Architecture Viz"], "#8B5CF6",
         os.path.join("assets", "banners", "lstm_module_banner_1774322380000_1774328471585.png")),
    ]

    st.markdown('<div id="modules-list"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-family: \'Montserrat\', sans-serif; margin-bottom: 25px; color: white; font-weight: 700; border-bottom: 2px solid #3B82F6; display: inline-block; padding-bottom: 10px;">Modules</h3>', unsafe_allow_html=True)
    
    for i, (ic, title, sub, desc, page, tags, clr, img) in enumerate(CARDS):
        with st.container():
            e_col1, e_col2, e_col3 = st.columns([1.2, 3, 1])
            with e_col1: st.image(img, width="stretch")
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

# ─────────────────────────────────────────────────────────────────────────────
# DEFINE PAGES
# ─────────────────────────────────────────────────────────────────────────────
p_home = st.Page(home_page,                     title="Dashboard",               icon="🧠", default=True)
p_pct  = st.Page("Perceptron/perceptron_ui.py", title="1. The Perceptron",       icon="🟢")
p_fwd  = st.Page("Forward_Propagation/forward_propagation.py", title="2. Forward Propagation",  icon="➡️")
p_bwd  = st.Page("Backward_Propagation/backward_propagation.py",title="3. Backward Propagation", icon="⬅️")
p_hop  = st.Page("Hopefield/hopefield.py",      title="4. Hopfield Network",     icon="🧠")

p_cv   = st.Page("OpenCV_Detection/page_gallery.py",    title="5. OpenCV Detection",     icon="📷")
p_cv_att = st.Page("OpenCV_Detection/page_attendance.py", title="5.1 CV Attendance", icon="📋")
p_cv_face = st.Page("OpenCV_Detection/page_face_scan.py", title="5.2 CV Face Scanner", icon="🔍")
p_cv_vehicle = st.Page("OpenCV_Detection/page_vehicle.py", title="5.3 CV Vehicles", icon="🚗")
p_cv_sign = st.Page("OpenCV_Detection/page_sign.py", title="5.4 CV Sign Detection", icon="🛑")
p_cv_palm = st.Page("OpenCV_Detection/page_palm.py", title="5.5 CV Palm Reading", icon="🖐️")

p_sa   = st.Page("Sentiment_Analysis/sentiment_analysis.py",  title="6. Sentiment Analysis",   icon="💬")
p_hub  = st.Page("LSTM_Application/lstm_hub.py", title="7. LSTM Application Hub", icon="🚀")

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
pages = {
    "Home":              [p_home],
    "Neural Networks":   [p_pct, p_fwd, p_bwd, p_hop],
    "OpenCV Lab":        [p_cv, p_cv_att, p_cv_face, p_cv_vehicle, p_cv_sign, p_cv_palm],
    "Applied AI & LSTM": [p_sa, p_hub],
}

pg = st.navigation(pages)

# ── LOGIC ──
try:
    if "last_visited_page" not in st.session_state:
        st.session_state.last_visited_page = pg.title
    
    if st.session_state.last_visited_page != pg.title:
        st.session_state.lstm_active_mod = None
        st.session_state.last_visited_page = pg.title
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("lstm_scrolled_")]
        for k in keys_to_clear:
            del st.session_state[k]

    pg.run()
except Exception as e:
    st.error(f"Error: {e}")
