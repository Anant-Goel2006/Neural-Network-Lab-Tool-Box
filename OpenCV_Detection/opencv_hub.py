import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
from utils.styles import section_header, gradient_header
import cv2

try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av
    WEBRTC_READY = True
except ImportError:
    WEBRTC_READY = False

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

import tempfile
import os

def process_video_realtime(video_file, callback_fn):
    """Processes an uploaded video file frame-by-frame with a callback."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    stop_btn = st.button("⏹ Stop Processing", key=f"stop_{video_file.name}")
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break
        
        # Process frame
        processed_frame = callback_fn(frame)
        
        # Display
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(0.01) # Small delay for UI stability
        
    cap.release()
    os.unlink(tfile.name)
    st.success("Video Processing Complete!")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — ATTENDANCE SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
def _attendance_module():
    section_header("Attendance System", "Auto-detect & log multiple faces · CSV export")
    if "cv_attendance" not in st.session_state: st.session_state.cv_attendance=[]

    c1,c2=st.columns([1,1])
    with c1:
        reg_name=st.text_input("Full Name (Target)", placeholder="e.g. Clark Kent", key="cv_reg_name")
        reg_id=st.text_input("ID / Roll No.", placeholder="e.g. DC-001", key="cv_reg_id")
        
        st.divider()
        src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_att_src")
        
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        def _att_cb(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            for idx, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 91, 234), 3)
                person = reg_name if reg_name else f"Person {idx+1}"
                cv2.putText(img, person, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 91, 234), 2)
            return img

        if src == "📷 Photo":
            f = st.file_uploader("Upload Target Photo", type=["jpg", "png"], key="cv_att_photo")
            if f and st.button("📸 Detect & Register", type="primary", use_container_width=True):
                img_bytes = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                processed = _att_cb(img)
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

        elif src == "📹 Video File":
             v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="cv_att_video")
             if v: process_video_realtime(v, _att_cb)

        else:
            st.markdown("**Live WebRTC Camera**")
            def face_log_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
                
                for idx, (x,y,w,h) in enumerate(faces):
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 91, 234), 3) # Superman Blue
                    person = reg_name if reg_name else f"Person {idx+1}"
                    cv2.putText(img, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 91, 234), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(key="att_stream", video_frame_callback=face_log_callback, rtc_configuration=RTC_CONFIG, media_stream_constraints={"video": True, "audio": False})

    with c2:
        section_header("Attendance Log", f"{len(st.session_state.cv_attendance)} entries")
        st.info("Log your attendance manually after detection.")
        if st.button("Log Current Detection"):
            st.session_state.cv_attendance.append({
                "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "Name": reg_name or "Unknown", "ID": reg_id or "N/A", "Status": "Present"
            })
            st.rerun()

        if st.session_state.cv_attendance:
            df = pd.DataFrame(st.session_state.cv_attendance)
            st.dataframe(df, hide_index=True, use_container_width=True)
            if st.button("🗑 Clear Log", use_container_width=True):
                st.session_state.cv_attendance=[]; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FACE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
def _face_scan_module():
    section_header("Face Scanner", "Multi-face · Eyes · Smile")
    
    src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_fs_src")
    
    face_cas = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    eye_cas = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
    smile_cas = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")

    def _fs_cb(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (242, 169, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cas.detectMultiScale(roi_gray, 1.1, 5)
            for (ex,ey,ew,eh) in eyes: cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,177,64), 2)
            smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)
            for (sx,sy,sw,sh) in smiles: cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (237, 29, 36), 2)
        return img

    if src == "📷 Photo":
        f = st.file_uploader("Upload Photo", type=["jpg", "png"], key="fs_photo")
        if f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            processed = _fs_cb(img)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="fs_video")
        if v: process_video_realtime(v, _fs_cb)
    else:
        def face_scan_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(242, 169, 0),3) # Flash Yellow
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cas.detectMultiScale(roi_gray, 1.1, 5)
                for (ex,ey,ew,eh) in eyes: cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,177,64),2)
                smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)
                for (sx,sy,sw,sh) in smiles: cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(237, 29, 36),2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.markdown("**Live WebRTC Camera**")
        webrtc_streamer(key="face_scan_stream", video_frame_callback=face_scan_callback, rtc_configuration=RTC_CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — VEHICLES (YOLO MOCK / CASCADE)
# ═════════════════════════════════════════════════════════════════════════════
def _vehicle_module():
    section_header("Vehicle Detection", "YOLOv8 Real-Time Counting")
    
    src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_vd_src")
    
    # Cache model to avoid re-downloading
    if 'yolo_model' not in st.session_state:
        try:
            from ultralytics import YOLO
            with st.spinner("Loading YOLOv8n (Weights)..."):
                st.session_state.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            st.error(f"Failed to load YOLO: {e}")
            return
    model = st.session_state.yolo_model

    def _vd_cb(img):
        results = model(img, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            name = results.names[cls]
            if name in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 177, 64), 3)
                cv2.putText(img, f"{name.upper()} {conf:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 177, 64), 2)
        return img

    if src == "📷 Photo":
        f = st.file_uploader("Upload Image", type=["jpg", "png"], key="vd_photo")
        if f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            processed = _vd_cb(img)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="vd_video")
        if v: process_video_realtime(v, _vd_cb)
    else:
        st.info("Live YOLOv8 Inference Running...")
        try:
            from ultralytics import YOLO
            # Cache model to avoid re-downloading
            if 'yolo_model' not in st.session_state:
                with st.spinner("Downloading YOLOv8n (Weights)..."):
                    st.session_state.yolo_model = YOLO('yolov8n.pt')
            model = st.session_state.yolo_model
        except Exception as e:
            st.error(f"YOLO Error: {e}")
            return

        def vehicle_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = model(img, verbose=False)[0]
            # Draw boxes for vehicles/people
            for box in results.boxes:
                cls = int(box.cls[0])
                name = results.names[cls]
                if name in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 177, 64), 3)
                    cv2.putText(img, f"{name.upper()} {conf:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 177, 64), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.markdown("**Live WebRTC Camera (YOLOv8)**")
        webrtc_streamer(key="vehicle_stream", video_frame_callback=vehicle_callback, rtc_configuration=RTC_CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — TRAFFIC SIGNS
# ═════════════════════════════════════════════════════════════════════════════
def _sign_module():
    section_header("Traffic Sign Detection", "CNN Classifier · 43 Classes")
    src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_sd_src")
    
    def _sd_cb(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
        m2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))
        red_mask = cv2.bitwise_or(m1, m2)
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 1000:
                rect = cv2.minAreaRect(c)
                box = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(img, [box], 0, (237, 29, 36), 3) 
                cv2.putText(img, "Red Sign", (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (237, 29, 36), 2)
        return img

    if src == "📷 Photo":
        f = st.file_uploader("Upload Sign", type=["jpg", "png"], key="sd_photo")
        if f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            processed = _sd_cb(img)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="sd_video")
        if v: process_video_realtime(v, _sd_cb)
    else:
        def sign_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
            m2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))
            red_mask = cv2.bitwise_or(m1, m2)
            cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    rect = cv2.minAreaRect(c)
                    box = np.intp(cv2.boxPoints(rect))
                    cv2.drawContours(img, [box], 0, (237, 29, 36), 3) 
                    cv2.putText(img, "Red Sign", (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (237, 29, 36), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.markdown("**Live WebRTC Camera**")
        webrtc_streamer(key="sign_stream", video_frame_callback=sign_callback, rtc_configuration=RTC_CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PALM READING (MediaPipe Hand Tracking)
# ═════════════════════════════════════════════════════════════════════════════
def _palm_module():
    section_header("🖐️ Palm Reading", "Real-time hand skeleton tracking via MediaPipe")
    src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_palm_src")
    
    try:
        import mediapipe as mp
        from mediapipe.solutions import hands as mp_hands
        from mediapipe.solutions import drawing_utils as mp_drawing
        hands_eval = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    except Exception as e:
        st.markdown("""
            <div style="background:#FFF9E6; border:2px solid #FFCC00; padding:12px; border-radius:0px; margin-bottom:20px;">
                <div style="font-weight:900; color:#856404; font-size:14px;">⚠️ MODULE UNAVAILABLE (ENVIRONMENT)</div>
                <div style="font-size:12px; color:#856404; margin-top:4px;">
                    Hand tracking requires MediaPipe, which is currently incompatible with Python 3.13 on your system.
                </div>
            </div>
        """, unsafe_allow_html=True)
        with st.expander("🛠️ Why am I seeing this?"):
            st.info("The MediaPipe package (0.10.x) lacks the mandatory 'solutions' module in the official Python 3.13 builds for Windows. To use this feature, consider using a Python 3.11 or 3.12 environment.")
        return

    def _palm_cb(img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_eval.process(rgb_img)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
        return img

    if src == "📷 Photo":
        f = st.file_uploader("Upload Hand Photo", type=["jpg", "png"], key="palm_photo")
        if f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            processed = _palm_cb(img)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="palm_video")
        if v: process_video_realtime(v, _palm_cb)
    else:
        def palm_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands_eval.process(rgb_img)
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        st.markdown("**Live WebRTC Camera**")
        webrtc_streamer(key="palm_stream", video_frame_callback=palm_callback, rtc_configuration=RTC_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def opencv_detection_page():
    from utils.styles import inject_global_css
    inject_global_css()
    gradient_header("OpenCV Detection Lab",
        "Face Recognition · YOLO Vehicles · Traffic Signs · Attendance · Palm Reading", "👁️")

    if not WEBRTC_READY:
        st.error("`streamlit-webrtc` is missing. Features requiring live camera will not function.")

    # ── Module Selector ─────────────────────────────────────────────────────
    MODULES = [
        ("📋","Attendance","Face log · CSV","attendance","#005BEA"),
        ("🔍","Face Scanner","Eyes · Smile","face_scan","#F2A900"),
        ("🚗","Vehicles","Traffic · Live Analytics","vehicle","#00B140"),
        ("🛑","Sign Detection","Shapes · Colors","sign","#ED1D24"),
        ("🖐️","Palm Reading","Hands · Gestures","palm","#404040"),
    ]
    cols = st.columns(5)
    for i,(icon,title,desc,key,clr) in enumerate(MODULES):
        active = st.session_state.get("cv_module","attendance")==key
        border = f"{clr}" if active else "#000"
        cols[i].markdown(f"""
        <div style="background-color:#FFF; border:3px solid {border};
            box-shadow: 4px 4px 0px {clr if active else '#1212'};
            border-radius:0px; padding:12px 10px; text-align:center;
            transition:all 0.2s; height: 110px; display: flex; flex-direction:column; justify-content:center;
            cursor: pointer; margin-bottom: 10px;">
            <div style="font-size:28px; margin-bottom:4px;">{icon}</div>
            <div style="font-family:'Impact', sans-serif; font-size:16px; color:{'#121212'}; text-transform:uppercase;">{title}</div>
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-top:2px; text-transform:uppercase;">{desc}</div>
        </div>""", unsafe_allow_html=True)
        if cols[i].button(f"Open {title}", key=f"cv_btn_{key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.cv_module=key; st.rerun()

    mod = st.session_state.get("cv_module","attendance")
    st.divider()

    if mod == "attendance": _attendance_module()
    elif mod == "face_scan": _face_scan_module()
    elif mod == "vehicle": _vehicle_module()
    elif mod == "sign": _sign_module()
    elif mod == "palm": _palm_module()
