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
p_sa   = st.Page(sentiment_analysis_page,  title="5. NLP Neural Network",   icon="🤖")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR BRANDING
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_brand():
    st.sidebar.markdown("""
    <div style="padding:24px 16px 22px;text-align:center;
        margin-bottom:12px; background:rgba(10, 10, 20, 0.9); border:3px solid #1e1b4b; border-bottom: 4px solid #00f0ff; border-radius: 0px; box-shadow: 5px 5px 0px #8b5cf6; word-wrap: break-word;">
        <div style="font-size:48px;margin-bottom:12px; text-shadow: 3px 3px 0px #8b5cf6;">🦸</div>
        <div style="font-family:'Oswald',sans-serif; font-size:24px;font-weight:700;letter-spacing:2px;
            color:#FAFAFA; text-transform: uppercase; text-shadow: 2px 2px 0px #00f0ff;">NEUROLAB</div>
        <div style="font-size:12px;color:#00f0ff;letter-spacing:2px;
            margin-top:6px;text-transform:uppercase;font-weight:600; font-family:'Inter', sans-serif;">
            Cosmic Graphic Edition
        </div>
        <div style="display:flex;justify-content:center;gap:8px;margin-top:20px; flex-wrap:wrap;">
            <span style="background:#8b5cf6;border:2px solid #3b0764;
                color:#FAFAFA;font-size:10px;padding:4px 10px;font-weight:700; font-family:'Oswald';
                letter-spacing:1px; text-transform: uppercase; box-shadow: 2px 2px 0px #040014;">v6.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home_page():
    inject_global_css()

    st.markdown("""
    <div style="position: relative; overflow: hidden; text-align:center; padding: 60px 20px; background: rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b; border-bottom: 6px solid #8b5cf6; margin-bottom: 30px; box-shadow: 8px 8px 0px #00f0ff; word-wrap: break-word; overflow-wrap: break-word;">
        <div style="font-size: 80px; margin-bottom: 16px; text-shadow: 4px 4px 0px #8b5cf6;">💥</div>
        <h1 style="font-size: 64px; font-weight: 700; color: #FAFAFA; margin: 0; font-family:'Oswald', sans-serif; letter-spacing: 4px; text-shadow: 5px 5px 0px #8b5cf6, -2px -2px 0px #00f0ff; text-transform: uppercase;">NEUROLAB</h1>
        <div style="width: 100px; height: 5px; background: #00f0ff; margin: 24px auto; box-shadow: 3px 3px 0px #3b0764;"></div>
        <p style="font-size: 18px; color: #a78bfa; font-family: 'Oswald', sans-serif; max-width: 800px; margin: 16px auto 0; line-height: 1.6; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; background:#040014; padding: 5px 15px; border: 2px solid #1e1b4b; display:inline-block;">
            Comic Universe Matrix Activated
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
    st.markdown("""<div style="display:flex;align-items:center;gap:14px;margin:30px 0;">
        <div style="flex:1;height:2px;background:#27272A;"></div>
        <span style="font-size:16px;color:#A1A1AA;letter-spacing:4px;
            font-weight:500;text-transform:uppercase; font-family:'Oswald', sans-serif;">Lab Modules</span>
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

        ("👁️", "OpenCV Detection", "Vision Lab",
         "4 sub-modules: <strong>Attendance</strong>, <strong>Face Scanner</strong>, <strong>Vehicle Detection</strong> with live counting analytics, and <strong>Sign Detection</strong>.",
         p_cv, ["YOLO Vehicles","Live Analytics","Face Scanner","CSV Export"], "#F59E0B"),

        ("🤖", "NLP Neural Network", "LSTM Language Processor",
         "8-emotion natural language processing with <strong>mixed sentiment detection</strong>, word-level highlights, radar charts, and batch mode.",
         p_sa, ["8 Emotions","LSTM Core","Mixed Sentiments","Batch Mode"], "#EC4899"),
    ]

    def _card(ic, title, sub, desc, tags, clr):
        tags_html = "".join([
            f'<span style="background:#040014; border: 2px solid {clr}; color:{clr};'
            f'font-size:11px; padding:4px 10px; font-weight:700; font-family: Oswald;'
            f'margin-right:6px; display:inline-block; margin-bottom:6px; text-transform: uppercase; letter-spacing: 1px;">{t}</span>' for t in tags
        ])
        return f"""<div style="background:rgba(10, 10, 20, 0.85); backdrop-filter: blur(8px); border: 3px solid #1e1b4b;
            padding:24px; min-height: 380px; box-shadow: 6px 6px 0px {clr};
            transition:all 0.2s; word-wrap: break-word; overflow-wrap: break-word; display: flex; flex-direction: column;"
            onmouseover="this.style.boxShadow='9px 9px 0px rgba(0,240,255,1)'; this.style.borderColor='{clr}'; this.style.transform='translate(-3px, -3px)';"
            onmouseout="this.style.boxShadow='6px 6px 0px {clr}'; this.style.borderColor='#1e1b4b'; this.style.transform='translate(0, 0)';">
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px; flex-wrap:wrap;">
                <div style="font-size:48px; text-shadow: 3px 3px 0px {clr};">{ic}</div>
                <div style="flex:1; min-width:150px;">
                    <div style="font-size:26px; font-weight:700; color:#FAFAFA; font-family:'Oswald', sans-serif; letter-spacing:2px; text-transform: uppercase; line-height:1.2;">{title}</div>
                    <div style="font-size:14px; color:{clr}; font-weight:600; letter-spacing: 2px; text-transform:uppercase; margin-top:4px; font-family:'Oswald';">{sub}</div>
                </div>
            </div>
            <div style="font-size:15px; color:#E4E4E7; line-height:1.6; font-family:'Inter'; margin-bottom:20px; font-weight:400; flex-grow:1;">{desc}</div>
            <div style="display:flex; flex-wrap:wrap;">{tags_html}</div>
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
            font-weight:500;text-transform:uppercase; font-family:'Oswald', sans-serif;">Core Tech Stack</span>
        <div style="flex:1;height:2px;background:#27272A;"></div>
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
