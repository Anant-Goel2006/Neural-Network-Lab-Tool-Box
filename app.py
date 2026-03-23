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

    st.markdown("""
    <div class="fade-in" style="text-align:center; padding: 100px 40px; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6); margin-bottom: 60px; position: relative; overflow: hidden; backdrop-filter: blur(25px);">
        <div style="position: absolute; top:0; left:0; right:0; height: 1px; background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);"></div>
        <div style="font-size: 120px; margin-bottom: 30px; filter: drop-shadow(0 0 30px rgba(59, 130, 246, 0.4)); position:relative; z-index:2; animation: float 6s ease-in-out infinite;">🧠</div>
        <h1 style="font-size: 90px; color: #F8FAFC; margin: 0; line-height: 1; font-weight: 800; font-family: 'Montserrat', sans-serif; letter-spacing: 4px; position:relative; z-index:2; text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);">NEUROLAB</h1>
        <div style="font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 600; color: #60A5FA; margin-top: 30px; text-transform: uppercase; letter-spacing: 10px; position:relative; z-index:2; display:inline-block; background: rgba(59, 130, 246, 0.1); padding: 8px 20px; border-radius: 30px; border: 1px solid rgba(59, 130, 246, 0.2);">
            THE ULTIMATE NEURAL FRONTIER
        </div>
    </div>
    <style>
        @keyframes float { 0% {transform: translateY(0px);} 50% {transform: translateY(-20px);} 100% {transform: translateY(0px);} }
        .fade-in { animation: fadeIn 1.2s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.info("**Engine:** NumPy + OpenCV")
    c2.success("**UI Paradigm:** Native Safe Dark Mode")
    c3.warning("**NLP Core:** O(1) Lexicon")

    st.divider()

    # ── Stats Row ───────────────────────────────────────────────────────────
    s_cols = st.columns(5)
    s_cols[0].metric("Modules", "5", "🧩 Built-in")
    s_cols[1].metric("Architectures", "∞", "🏗️ Custom")
    s_cols[2].metric("Emotions", "8", "💭 Tracked")
    s_cols[3].metric("CV Modes", "5", "📷 Live Cam")
    s_cols[4].metric("Interactivity", "100%", "⚡ Real-time")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section Divider ─────────────────────────────────────────────────────
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:30px 0;">
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        <span style="font-size:14px;color:#94A3B8;letter-spacing:4px;
            font-weight:600;text-transform:uppercase; font-family:'Inter', sans-serif;">Lab Modules</span>
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
    </div>""", unsafe_allow_html=True)

    # ── Module Cards ────────────────────────────────────────────────────────
    CARDS = [
        ("🟢", "The Perceptron", "Binary Classifier",
         "Train a single neuron with <strong>live convergence animation</strong>. Watch the decision boundary shift epoch-by-epoch with real-time speedometer gauges.",
         p_pct, ["Live Training","6 Gates","Confusion Matrix","Weight Trajectory"], "#10B981"),

        ("➡️", "Forward Propagation", "Signal Flow Engine",
         "Trace inputs through every layer. Build custom architectures with <strong>6 activations</strong>, 4 loss functions, and view live activation heatmaps.",
         p_fwd, ["6 Activations","Layer Heatmap","Custom Arch","Weight Editor"], "#06B6D4"),

        ("⬅️", "Backward Propagation", "Gradient Engine",
         "Watch gradients flow backwards via chain rule. Visualize <strong>vanishing & exploding gradients</strong> with live bar charts and heatmaps.",
         p_bwd, ["Chain Rule Viz","Gradient Heatmap","Weight Diff","Live Gauges"], "#8B5CF6"),

        ("📷", "OpenCV Detection", "Vision Lab",
         "4 sub-modules: <strong>Attendance</strong>, <strong>Face Scanner</strong>, <strong>Vehicle Detection</strong> with live counting analytics, and <strong>Sign Detection</strong>.",
         p_cv, ["YOLO Vehicles","Live Analytics","Face Scanner","CSV Export"], "#F59E0B"),

        ("💬", "Sentiment Analysis", "LSTM Language Processor",
         "8-emotion natural language processing with <strong>mixed sentiment detection</strong>, word-level highlights, radar charts, and batch mode.",
         p_sa, ["8 Emotions","LSTM Core","Mixed Sentiments","Batch Mode"], "#EC4899"),
    ]

    def _card(ic, title, sub, desc, tags, clr):
        tags_html = "".join([
            f'<span style="background: rgba({int(clr[1:3], 16)}, {int(clr[3:5], 16)}, {int(clr[5:7], 16)}, 0.1); color: {clr};'
            f'font-size: 11px; padding: 4px 10px; font-weight: 600; font-family: \'Inter\', sans-serif; border: 1px solid rgba({int(clr[1:3], 16)}, {int(clr[3:5], 16)}, {int(clr[5:7], 16)}, 0.2);'
            f'border-radius: 20px; margin-right: 8px; display: inline-block; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">{t}</span>' for t in tags
        ])
        return f"""<div class="premium-card fade-in" style="min-height: 420px; display: flex; flex-direction: column;">
            <div style="display:flex;align-items:flex-start;gap:20px;margin-bottom:24px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom:20px; position:relative;">
                <div style="font-size:44px; filter: drop-shadow(0 0 15px rgba({int(clr[1:3], 16)}, {int(clr[3:5], 16)}, {int(clr[5:7], 16)}, 0.4)); z-index:2; background: rgba(15,23,42,0.8); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);">{ic}</div>
                <div style="flex:1; z-index:2; padding-top: 6px;">
                    <div style="font-size:26px; font-family:'Montserrat', sans-serif; font-weight: 700; color:#F8FAFC; line-height:1.2; margin-bottom: 6px; letter-spacing: 0.5px;">{title}</div>
                    <div style="font-size:13px; font-family:'Inter', sans-serif; font-weight: 600; color:{clr}; text-transform:uppercase; letter-spacing: 1.5px; opacity: 0.9;">{sub}</div>
                </div>
                <div style="position:absolute; top: -15px; right: -15px; width:80px; height:80px; background: {clr}; opacity:0.12; border-radius:50%; filter:blur(30px);"></div>
            </div>
            <div style="font-size:15px; color:#CBD5E1; line-height:1.7; font-family:'Inter', sans-serif; margin-bottom:30px; font-weight:400; flex-grow:1;">{desc}</div>
            <div style="display:flex; flex-wrap:wrap; margin-bottom:10px; gap: 8px;">{tags_html}</div>
        </div>"""

    # Row 1
    r1 = st.columns(3)
    for i in range(3):
        ic, title, sub, desc, page, tags, clr = CARDS[i]
        with r1[i]:
            st.markdown(_card(ic, title, sub, desc, tags, clr), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(f"Open {title} →", key=f"home_{i}", width="stretch", type="primary"):
                st.switch_page(page)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    r2 = st.columns(2)
    for i in range(2):
        ic, title, sub, desc, page, tags, clr = CARDS[3+i]
        with r2[i]:
            st.markdown(_card(ic, title, sub, desc, tags, clr), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(f"Open {title} →", key=f"home_{3+i}", width="stretch", type="primary"):
                st.switch_page(page)

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
