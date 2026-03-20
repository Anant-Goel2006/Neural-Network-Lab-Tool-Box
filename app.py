import streamlit as st

st.set_page_config(
    page_title="Neural Network Lab",
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
p_cv   = st.Page(opencv_detection_page,    title="4. OpenCV Detection",     icon="👁️")
p_sa   = st.Page(sentiment_analysis_page,  title="5. Sentiment Analysis",   icon="💬")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR BRANDING
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_brand():
    st.sidebar.markdown("""
    <div style="padding:24px 16px 22px;text-align:center;
        border-bottom:2px solid #121212;margin-bottom:12px; background:#FFF; border:3px solid #121212; box-shadow: 4px 4px 0px #005BEA;">
        <div style="font-size:36px;margin-bottom:8px;">🧠</div>
        <div style="font-family:'Bangers',sans-serif; font-size:24px;font-weight:900;letter-spacing:2px;
            color:#121212;">NEURAL LAB</div>
        <div style="font-size:10px;color:#64748B;letter-spacing:2px;
            margin-top:6px;text-transform:uppercase;font-weight:900;">
            Deep Learning Toolbox
        </div>
        <div style="display:flex;justify-content:center;gap:5px;margin-top:14px;">
            <span style="background:#ED1D24;border:1px solid #121212;
                color:#FFFFFF;font-size:9px;padding:2px 8px;font-weight:800;
                letter-spacing:1px;box-shadow:2px 2px 0px #121212;">v2.5</span>
            <span style="background:#F2A900;border:1px solid #121212;
                color:#121212;font-size:9px;padding:2px 8px;font-weight:800;
                letter-spacing:1px;box-shadow:2px 2px 0px #121212;">5 MODULES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home_page():
    inject_global_css()

    st.markdown("""
    <div style="text-align:center; padding: 40px 0;">
        <div style="font-size: 80px; margin-bottom: 20px;">🧠</div>
        <h1 style="font-size: 56px; font-weight: 900; color: #121212; margin: 0; font-family:'Bangers',cursive;">Neural Network Lab</h1>
        <p style="font-size: 20px; color: #64748B; max-width: 700px; margin: 20px auto 0; line-height: 1.6; font-weight: 700;">
            THE ULTIMATE SUPER-POWERED TOOLKIT FOR DEEP LEARNING VISUALIZATION, LIVE OPENCV ANALYTICS, AND LSTM PROCESSING.
        </p>
    </div>
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
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:20px 0 30px;">
        <div style="flex:1;height:3px;background:#121212;"></div>
        <span style="font-size:12px;color:#121212;letter-spacing:4px;
            font-weight:900;text-transform:uppercase; font-family:'Impact';">Super Modules</span>
        <div style="flex:1;height:3px;background:#121212;"></div>
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

        ("👁️", "OpenCV Detection", "Vision Lab",
         "4 sub-modules: <strong>Attendance</strong>, <strong>Face Scanner</strong>, <strong>Vehicle Detection</strong> with live counting analytics, and <strong>Sign Detection</strong>.",
         p_cv, ["YOLO Vehicles","Live Analytics","Face Scanner","CSV Export"], "#F59E0B"),

        ("💬", "Sentiment Analysis", "NLP Engine",
         "8-emotion analysis with <strong>mixed sentiment detection</strong>, word-level highlights, radar charts, batch mode, and text comparison.",
         p_sa, ["8 Emotions","Word Highlights","Comparison","Batch Mode"], "#EC4899"),
    ]

    def _card(ic, title, sub, desc, tags, clr):
        tags_html = "".join([
            f'<span style="background:#FFFFFF;border:2px solid #121212;color:#121212;'
            f'font-size:9px;padding:3px 9px;font-weight:900;'
            f'letter-spacing:1px; box-shadow: 2px 2px 0px #121212; margin-right:4px; text-transform:uppercase;">{t}</span>' for t in tags
        ])
        return f"""<div style="background:#FFFFFF; border:3px solid #121212;
            border-top:8px solid {clr}; border-radius:0px; padding:24px 22px; height:100%;
            box-shadow: 6px 6px 0px #121212; transition:all 0.2s;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                <div style="font-size:36px;">{ic}</div>
                <div>
                    <div style="font-size:20px; font-weight:900; color:#121212; font-family:'Bangers'; letter-spacing:1px;">{title.upper()}</div>
                    <div style="font-size:11px; color:{clr}; letter-spacing:2px; text-transform:uppercase;
                        margin-top:2px; font-weight:900;">{sub}</div>
                </div>
            </div>
            <div style="font-size:14px; color:#121212; line-height:1.6; font-weight:700; margin-bottom:16px;">{desc}</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">{tags_html}</div>
        </div>"""

    # Row 1
    r1 = st.columns(3)
    for i in range(3):
        ic, title, sub, desc, page, tags, clr = CARDS[i]
        with r1[i]:
            st.markdown(_card(ic, title, sub, desc, tags, clr), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(f"Open {title} →", key=f"home_{i}", use_container_width=True, type="primary"):
                st.switch_page(page)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    r2 = st.columns(2)
    for i in range(2):
        ic, title, sub, desc, page, tags, clr = CARDS[3+i]
        with r2[i]:
            st.markdown(_card(ic, title, sub, desc, tags, clr), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(f"Open {title} →", key=f"home_{3+i}", use_container_width=True, type="primary"):
                st.switch_page(page)

    # ── Tech Stack ──────────────────────────────────────────────────────────
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:32px 0 20px;">
        <div style="flex:1;height:2px;background:#E2E8F0;"></div>
        <span style="font-size:10px;color:#64748B;letter-spacing:3px;
            font-weight:900;text-transform:uppercase;">Tech Stack</span>
        <div style="flex:1;height:2px;background:#E2E8F0;"></div>
    </div>""", unsafe_allow_html=True)

    STACK = [("Streamlit","#FF4B4B"),("NumPy","#005BEA"),("Plotly","#ED1D24"),
             ("Pandas","#121212"),("OpenCV","#005BEA"),("TensorFlow","#FF6F00"),
             ("MediaPipe","#ED1D24"),("WebRTC","#005BEA")]
    st.markdown('<div style="text-align:center;line-height:2.8;">' +
        " ".join([f'<span style="background:{c}12;border:2px solid {c};color:{c};'
                  f'font-size:11px;padding:4px 14px;font-weight:900;'
                  f'letter-spacing:1px;margin:2px;text-transform:uppercase;box-shadow:2px 2px 0px #121212;">{n}</span>' for n,c in STACK]) +
        '</div>', unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:24px;padding-bottom:12px;">
        <span style="font-size:12px;color:#64748B;font-weight:900;text-transform:uppercase;letter-spacing:1px;">
            Neural Network Lab · Powered by DC Justice Suite · v2.5 Deployment Ready
        </span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
p_home = st.Page(home_page, title="Dashboard", icon="🏠", default=True)

pages = {
    "Home":           [p_home],
    "Neural Network": [p_pct, p_fwd, p_bwd],
    "Applied AI":     [p_cv, p_sa],
}

inject_global_css()
sidebar_brand()
pg = st.navigation(pages)
pg.run()
