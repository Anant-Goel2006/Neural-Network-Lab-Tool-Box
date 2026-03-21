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
    <div style="padding:25px; text-align:center; background:#020617; border:4px solid #000; box-shadow: 6px 6px 0px #EF4444; margin-bottom:30px;">
        <div style="font-size:60px; margin-bottom:15px; filter: drop-shadow(3px 3px 0px #000);">🧠</div>
        <div style="font-family:'Bangers', cursive; font-size:36px; color:#FFFFFF; letter-spacing:3px; text-shadow: 2px 2px 0px #000;">NEUROLAB</div>
        <div style="font-family:'Luckiest Guy', cursive; font-size:14px; color:#FACC15; margin-top:5px; text-transform:uppercase; letter-spacing:1px;">
            Cosmic Edition v5.0
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home_page():
    inject_global_css()

    st.markdown("""
    <div style="text-align:center; padding: 100px 40px; background: #1e1b4b; border: 6px solid #000; box-shadow: 15px 15px 0px #EF4444; margin-bottom: 60px; position: relative; overflow: hidden;">
        <div style="position: absolute; top:0; left:0; width:100%; height:100%; background: radial-gradient(#ffffff11 1px, transparent 0); background-size: 8px 8px; opacity: 0.5;"></div>
        <div style="font-size: 120px; margin-bottom: 20px; filter: drop-shadow(5px 5px 0px #000);">🧠</div>
        <h1 style="font-size: 100px; color: #FFFFFF; margin: 0; line-height: 0.9;">NEUROLAB</h1>
        <div style="font-family: 'Luckiest Guy', cursive; font-size: 32px; color: #FACC15; margin-top: 20px; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 0px #000;">
            THE COSMIC NEURAL FRONTIER
        </div>
        <div style="width: 200px; height: 8px; background: #FFFFFF; margin: 30px auto; border: 3px solid #000;"></div>
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
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:30px 0;">
        <div style="flex:1;height:2px;background:#27272A;"></div>
        <span style="font-size:16px;color:#A1A1AA;letter-spacing:4px;
            font-weight:500;text-transform:uppercase; font-family:'Luckiest Guy', cursive;">Lab Modules</span>
        <div style="flex:1;height:2px;background:#27272A;"></div>
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
            f'<span style="background:#000; color:{clr};'
            f'font-size:12px; padding:5px 12px; font-weight:700; font-family: \'Luckiest Guy\', cursive; border: 1px solid {clr};'
            f'margin-right:8px; display:inline-block; margin-bottom:8px;">{t}</span>' for t in tags
        ])
        return f"""<div class="premium-card" style="min-height: 420px; display: flex; flex-direction: column;">
            <div style="display:flex;align-items:center;gap:20px;margin-bottom:25px; border-bottom: 4px solid #000; padding-bottom:15px;">
                <div style="font-size:50px; filter: drop-shadow(3px 3px 0px #000);">{ic}</div>
                <div style="flex:1;">
                    <div style="font-size:28px; font-family:'Bangers', cursive; color:#FFFFFF; line-height:1.1;">{title}</div>
                    <div style="font-size:14px; font-family:'Luckiest Guy', cursive; color:{clr}; text-transform:uppercase; margin-top:4px;">{sub}</div>
                </div>
            </div>
            <div style="font-size:17px; color:#F1F5F9; line-height:1.4; font-family:'Inter', sans-serif; margin-bottom:30px; font-weight:500; flex-grow:1;">{desc}</div>
            <div style="display:flex; flex-wrap:wrap; margin-bottom:10px;">{tags_html}</div>
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
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:40px 0 20px;">
        <div style="flex:1;height:2px;background:#27272A;"></div>
        <span style="font-size:12px;color:#A1A1AA;letter-spacing:3px;
            font-weight:500;text-transform:uppercase; font-family:'Luckiest Guy', cursive;">Core Tech Stack</span>
        <div style="flex:1;height:2px;background:#27272A;"></div>
    </div>""", unsafe_allow_html=True)

    STACK = [("Streamlit","#FF4B4B"),("NumPy","#005BEA"),("Plotly","#ED1D24"),
             ("Pandas","#121212"),("OpenCV","#005BEA"),("TensorFlow","#FF6F00"),
             ("MediaPipe","#ED1D24"),("WebRTC","#005BEA")]
    st.markdown('<div style="text-align:center;line-height:2.8;">' +
        " ".join([f'<span style="background:#1E293B; border:1px solid #334155; color:#94A3B8;'
                  f'font-size:12px; padding:6px 14px; font-weight:600;'
                  f'letter-spacing:1px; margin:4px; text-transform:uppercase; border-radius:4px;">{n}</span>' for n,c in STACK]) +
        '</div>', unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:40px;padding-bottom:20px;">
        <span style="font-size:13px; color:#64748B; font-weight:600; font-family:'Inter'; text-transform:uppercase; letter-spacing:0.05em;">
            NeuroLab Professional Suite © 2026 · Stable Release v4.2.0
        </span>
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
