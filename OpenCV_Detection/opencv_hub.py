from __future__ import annotations

import datetime
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils.palmistry_engine import (
    answer_palm_question,
    build_palm_report,
)
from utils.palmistry_knowledge import (
    build_professional_system_prompt,
    get_professional_greeting,
)
from utils.styles import section_header, gradient_header, render_content_card, render_info_grid, MODULE_THEMES, inject_module_theme

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av

    WEBRTC_READY = True
except ImportError:
    WEBRTC_READY = False

if WEBRTC_READY:
    RTC_CONFIG_LOCAL = RTCConfiguration({"iceServers": []})
    RTC_CONFIG_STUN = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        }
    )
else:
    RTC_CONFIG_LOCAL = None
    RTC_CONFIG_STUN = None

LIVE_INPUT_SOURCES = ["📷 Photo", "📸 Camera Snapshot", "🔴 Live WebRTC", "📹 Video File"]
LIVE_MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15, "max": 24},
    },
    "audio": False,
}

CV_GALLERY_PATH = "OpenCV_Detection/page_gallery.py"

CV_MODULES = [
    {
        "key": "attendance",
        "icon": "📋",
        "title": "Attendance",
        "gallery_subtitle": "Face log · Export CSV",
        "page_title": "OpenCV Attendance Studio",
        "page_subtitle": "Face logging, registration, and CSV-ready attendance tracking",
        "path": "OpenCV_Detection/page_attendance.py",
        "banner": os.path.join("assets", "banners", "attendance_banner_1774323273637.png"),
        "features": ["Real-time Face Detection", "User Registration", "CSV Attendance Export"],
    },
    {
        "key": "face_scan",
        "icon": "🔍",
        "title": "Face Scanner",
        "gallery_subtitle": "Eyes · Smile · ROI",
        "page_title": "OpenCV Face Scanner",
        "page_subtitle": "Multi-cascade face analysis with eyes, smile, and region tracking",
        "path": "OpenCV_Detection/page_face_scan.py",
        "banner": os.path.join("assets", "banners", "face_scanner_banner_1774323291585.png"),
        "features": ["Multi-Cascade Detection", "Ocular Tracking", "Mood and Smile Recognition"],
    },
    {
        "key": "vehicle",
        "icon": "🚗",
        "title": "Vehicles",
        "gallery_subtitle": "Traffic · Live Counting",
        "page_title": "OpenCV Vehicle Detection",
        "page_subtitle": "YOLO-powered traffic counting and live vehicle analytics",
        "path": "OpenCV_Detection/page_vehicle.py",
        "banner": os.path.join("assets", "banners", "vehicles_banner_1774323308501.png"),
        "features": ["YOLOv8 Inference", "Vehicle Classification", "Live Stats"],
    },
    {
        "key": "sign",
        "icon": "🛑",
        "title": "Sign Detection",
        "gallery_subtitle": "Shapes · Colors",
        "page_title": "OpenCV Sign Detection",
        "page_subtitle": "Color filtering and contour analysis for road-sign style recognition",
        "path": "OpenCV_Detection/page_sign.py",
        "banner": os.path.join("assets", "banners", "sign_detection_banner_1774323328063.png"),
        "features": ["Color Space Filtering", "Contour Analysis", "Symbolic Recognition"],
    },
    {
        "key": "palm",
        "icon": "🖐️",
        "title": "Palm Analysis",
        "gallery_subtitle": "60+ Features · Fine Lines · AI Analysis",
        "page_title": "Professional Palm Analyzer",
        "page_subtitle": "Advanced computer vision palm analysis with CLAHE, Gabor filters, and 60+ unique feature extraction",
        "path": "OpenCV_Detection/page_palm.py",
        "banner": os.path.join("assets", "banners", "palm_reading_banner_1774323346147.png"),
        "features": ["CNN + Gabor Detection", "60+ Feature Extraction", "Fine Line Analysis"],
    },
]

CV_MODULE_MAP = {item["key"]: item for item in CV_MODULES}


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
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), width="stretch")
        time.sleep(0.01) # Small delay for UI stability
        
    cap.release()
    os.unlink(tfile.name)
    st.success("Video Processing Complete!")


def _decode_image_file(file_obj):
    if file_obj is None:
        return None
    data = np.frombuffer(file_obj.getvalue(), np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _load_image_from_source(src, upload_label, upload_key, camera_label, camera_key):
    if src in ("📷 Photo", "📷 Photo Upload"):
        return _decode_image_file(st.file_uploader(upload_label, type=["jpg", "jpeg", "png"], key=upload_key))
    if src in ("📸 Camera Snapshot", "📸 Camera Auto-Scan"):
        return _decode_image_file(st.camera_input(camera_label, key=camera_key))
    return None


def _rtc_configuration_selector(key_prefix):
    if not WEBRTC_READY:
        return None

    st.caption(
        "Low-latency local mode is the default and avoids STUN. Turn STUN on only when the app is running remotely and the live stream will not connect."
    )
    use_stun = st.toggle("Use STUN servers", value=False, key=f"{key_prefix}_use_stun")
    return RTC_CONFIG_STUN if use_stun else RTC_CONFIG_LOCAL


def _start_webrtc_stream(stream_key, callback, label, key_prefix, hint=None):
    if not WEBRTC_READY:
        st.error("`streamlit-webrtc` is missing. Install it to enable live camera streaming.")
        return

    st.markdown(f"**{label}**")
    if hint:
        st.caption(hint)

    webrtc_streamer(
        key=stream_key,
        video_frame_callback=callback,
        rtc_configuration=_rtc_configuration_selector(key_prefix),
        media_stream_constraints=LIVE_MEDIA_STREAM_CONSTRAINTS,
        async_processing=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — ATTENDANCE SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
def _attendance_module():
    section_header("Face Log & Attendance", "Match Faces · Export CSV")
    # render_nlp_insight removed - now in chatbot
    if "cv_attendance" not in st.session_state: st.session_state.cv_attendance=[]

    c1,c2=st.columns([1,1])
    with c1:
        reg_name=st.text_input("Full Name (Target)", placeholder="e.g. Clark Kent", key="cv_reg_name")
        reg_id=st.text_input("ID / Roll No.", placeholder="e.g. DC-001", key="cv_reg_id")
        
        st.divider()
        src = st.radio("Input Source", LIVE_INPUT_SOURCES, horizontal=True, key="cv_att_src")
        
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        def _att_cb(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            for idx, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 91, 234), 3)
                person = reg_name if reg_name else f"Person {idx+1}"
                cv2.putText(img, person, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 91, 234), 2)
            return img

        if src in ("📷 Photo", "📸 Camera Snapshot"):
            img = _load_image_from_source(
                src,
                "Upload Target Photo",
                "cv_att_photo",
                "Capture Target Photo",
                "cv_att_camera",
            )
            if img is not None and st.button("📸 Detect & Register", type="primary", width="stretch"):
                processed = _att_cb(img.copy())
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch")

        elif src == "📹 Video File":
             v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="cv_att_video")
             if v: process_video_realtime(v, _att_cb)

        else:
            st.markdown("**Live WebRTC Camera**")
            def face_log_callback(frame: av.VideoFrame) -> av.VideoFrame:
                try:
                    # Frame skipping
                    if 'att_frame_count' not in st.session_state: st.session_state.att_frame_count = 0
                    st.session_state.att_frame_count += 1
                    if st.session_state.att_frame_count % 2 != 0: return frame
                    
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
                    
                    for idx, (x,y,w,h) in enumerate(faces):
                        cv2.rectangle(img, (x,y), (x+w, y+h), (239, 68, 68), 3) # Flash Red
                        person = reg_name if reg_name else f"TARGET {idx+1}"
                        cv2.putText(img, person.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239, 68, 68), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                except Exception:
                    return frame

            _start_webrtc_stream(
                "att_stream",
                face_log_callback,
                "Live WebRTC Camera",
                "att_stream",
                hint="If live streaming still fails in your setup, switch to Camera Snapshot above for a no-STUN fallback.",
            )

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
            st.dataframe(df, hide_index=True, width="stretch")
            if st.button("🗑 Clear Log", width="stretch"):
                st.session_state.cv_attendance=[]; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FACE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
def _face_scan_module():
    section_header("Face Scanner", "Multi-Cascade · Eyes & Smiles")
    
    src = st.radio("Input Source", LIVE_INPUT_SOURCES, horizontal=True, key="cv_fs_src")
    
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

    if src in ("📷 Photo", "📸 Camera Snapshot"):
        img = _load_image_from_source(
            src,
            "Upload Photo",
            "fs_photo",
            "Capture Photo",
            "fs_camera",
        )
        if img is not None:
            processed = _fs_cb(img.copy())
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch")
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="fs_video")
        if v: process_video_realtime(v, _fs_cb)
    else:
        def face_scan_callback(frame: av.VideoFrame) -> av.VideoFrame:
            try:
                # Frame skipping
                if 'fs_frame_count' not in st.session_state: st.session_state.fs_frame_count = 0
                st.session_state.fs_frame_count += 1
                if st.session_state.fs_frame_count % 2 != 0: return frame

                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(59, 130, 246),3) # Action Blue
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    eyes = eye_cas.detectMultiScale(roi_gray, 1.1, 5)
                    for (ex,ey,ew,eh) in eyes: cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(22, 197, 94),2) # Hulk Green
                    smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)
                    for (sx,sy,sw,sh) in smiles: cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(250, 204, 21),2) # Yellow
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame

        _start_webrtc_stream(
            "face_scan_stream",
            face_scan_callback,
            "Live WebRTC Camera",
            "face_scan_stream",
            hint="The stream runs at a lower default resolution for faster face detection.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — VEHICLES (YOLO MOCK / CASCADE)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Shared global cache for the YOLO detector."""
    try:
        from ultralytics import YOLO
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Failed to load YOLO: {e}")
        return None

def _vehicle_module():
    section_header("Vehicle Detection", "YOLOv8 Real-Time Counting")
    
    src = st.radio("Input Source", LIVE_INPUT_SOURCES, horizontal=True, key="cv_vd_src")
    
    model = load_yolo_model()
    if model is None: return

    def _vd_cb(img):
        results = model(img, verbose=False, imgsz=640)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            name = results.names[cls]
            if name in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 177, 64), 3)
                cv2.putText(img, f"{name.upper()} {conf:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 177, 64), 2)
        return img

    if src in ("📷 Photo", "📸 Camera Snapshot"):
        img = _load_image_from_source(
            src,
            "Upload Image",
            "vd_photo",
            "Capture Image",
            "vd_camera",
        )
        if img is not None:
            processed = _vd_cb(img.copy())
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch")
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="vd_video")
        if v: process_video_realtime(v, _vd_cb)
    else:
        st.info("Live YOLOv8 Inference Running...")
        model = load_yolo_model()
        if model is None: return

        def vehicle_callback(frame: av.VideoFrame) -> av.VideoFrame:
            try:
                # Frame skipping (YOLO is heavy)
                if 'vd_frame_count' not in st.session_state: st.session_state.vd_frame_count = 0
                st.session_state.vd_frame_count += 1
                if st.session_state.vd_frame_count % 3 != 0: return frame

                img = frame.to_ndarray(format="bgr24")
                results = model(img, verbose=False, imgsz=416)[0]
                for box in results.boxes:
                    cls = int(box.cls[0])
                    name = results.names[cls]
                    if name in ['car', 'truck', 'bus', 'motorcycle', 'person']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (239, 68, 68), 3) # Red
                        cv2.putText(img, f"{name.upper()} {conf:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame

        _start_webrtc_stream(
            "vehicle_stream",
            vehicle_callback,
            "Live WebRTC Camera (YOLOv8)",
            "vehicle_stream",
            hint="The detector uses lower-resolution live inference so it starts faster and drops fewer frames on CPU.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — TRAFFIC SIGNS
# ═════════════════════════════════════════════════════════════════════════════
def _sign_module():
    section_header("Traffic Sign Detection", "CNN Classifier · 43 Classes")
    
    src = st.radio("Input Source", LIVE_INPUT_SOURCES, horizontal=True, key="cv_sd_src")
    
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

    if src in ("📷 Photo", "📸 Camera Snapshot"):
        img = _load_image_from_source(
            src,
            "Upload Sign",
            "sd_photo",
            "Capture Sign",
            "sd_camera",
        )
        if img is not None:
            processed = _sd_cb(img.copy())
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch")
    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="sd_video")
        if v: process_video_realtime(v, _sd_cb)
    else:
        def sign_callback(frame: av.VideoFrame) -> av.VideoFrame:
            try:
                # Frame skipping
                if 'sd_frame_count' not in st.session_state: st.session_state.sd_frame_count = 0
                st.session_state.sd_frame_count += 1
                if st.session_state.sd_frame_count % 2 != 0: return frame

                img = frame.to_ndarray(format="bgr24")
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
                m2 = cv2.inRange(hsv, np.array( [160,100,100]), np.array([179,255,255]))
                red_mask = cv2.bitwise_or(m1, m2)
                cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) > 1000:
                        rect = cv2.minAreaRect(c)
                        box = np.intp(cv2.boxPoints(rect))
                        cv2.drawContours(img, [box], 0, (239, 68, 68), 10) 
                        cv2.putText(img, "SIGN DETECTED", (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 204, 21), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame

        _start_webrtc_stream(
            "sign_stream",
            sign_callback,
            "Live WebRTC Camera",
            "sign_stream",
            hint="Local mode starts without STUN to reduce camera handshake failures.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# PALM READING — ADVANCED COMPUTER VISION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Enhanced image preprocessing (CLAHE, Gabor, morphological)
# Phase 2: Fine line detection (branches, breaks, islands, forks, marks)
# Phase 3: Depth profiling along each line
# Phase 4: Spatial position analysis (start/end/gap measurements)
# Phase 5: 60+ unique features per palm
# ═════════════════════════════════════════════════════════════════════════════

def extract_skeleton(mask):
    mask_binary = (mask > 0).astype(np.uint8)
    return cv2.ximgproc.thinning(mask_binary * 255) if hasattr(cv2, 'ximgproc') else mask_binary


# ── PHASE 1: ENHANCED IMAGE PREPROCESSING (MULTI-SCALE) ───────────────────

def enhance_palm_image(image):
    """Multi-stage preprocessing to reveal even the finest, faintest palm lines.
    Pipeline: Multi-scale CLAHE → Unsharp mask → Bilateral denoise →
    Multi-scale Gabor bank (36 filters) → Canny edge fusion → LoG fusion →
    Adaptive threshold → Gentle morphological cleanup.
    Returns the enhanced binary line map and the CLAHE-enhanced grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Multi-scale CLAHE — reveal faint lines at different spatial scales
    clahe_fine = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    clahe_medium = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_coarse = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced_fine = clahe_fine.apply(gray)
    enhanced_medium = clahe_medium.apply(gray)
    enhanced_coarse = clahe_coarse.apply(gray)
    # Blend: take the maximum contrast across all scales
    enhanced = np.maximum(np.maximum(enhanced_fine, enhanced_medium), enhanced_coarse)

    # 2. Unsharp masking — sharpen fine crease details before detection
    gauss_blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    enhanced = cv2.addWeighted(enhanced, 1.5, gauss_blur, -0.5, 0)

    # 3. Bilateral filter — reduce noise while keeping line edges crisp
    denoised = cv2.bilateralFilter(enhanced, 9, 60, 60)

    # 4. Multi-scale Gabor filter bank — 4 scales × 12 orientations = 48 filters
    #    Captures everything from deep major lines to hair-thin faint creases
    gabor_responses = []
    scales = [
        (9,  1.5,  4.0, 0.4),  # Micro scale — ultra-faint intuition/intuition lines
        (15, 2.5,  6.0, 0.4),  # Fine scale — thin faint lines
        (21, 4.0, 10.0, 0.5),  # Medium scale — standard lines
        (31, 5.5, 14.0, 0.5),  # Coarse scale — deep major lines
    ]
    for ksize, sigma, lambd, gamma in scales:
        for theta_deg in range(0, 180, 15):  # 12 orientations (every 15°)
            theta = np.deg2rad(theta_deg)
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(denoised, cv2.CV_32F, kernel)
            gabor_responses.append(np.abs(filtered))

    # Max response across all orientations and scales
    gabor_max = np.max(np.stack(gabor_responses, axis=0), axis=0)
    gabor_norm = cv2.normalize(gabor_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. Canny edge detection — catches fine edges Gabor might miss
    canny_edges = cv2.Canny(denoised, 25, 80)

    # 6. Laplacian of Gaussian — catches thin crease-like structures
    log_input = cv2.GaussianBlur(denoised, (5, 5), 0)
    log_response = cv2.Laplacian(log_input, cv2.CV_64F)
    log_norm = cv2.normalize(np.abs(log_response), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 7. Fuse all detection layers — any method that finds a line keeps it
    fused = np.maximum(gabor_norm, canny_edges)
    fused = np.maximum(fused, log_norm)

    # 8. Adaptive threshold — handles uneven lighting across the palm (more sensitive)
    line_map = cv2.adaptiveThreshold(
        fused, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -3,
    )

    # 9. Morphological cleanup — connect broken segments first, then gentle noise removal
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_map = cv2.morphologyEx(line_map, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    line_map = cv2.morphologyEx(line_map, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 10. Remove only the tiniest noise specks (ultra-low threshold to preserve faint lines)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_map, connectivity=8)
    min_area = max(3, line_map.shape[0] * line_map.shape[1] * 0.00002)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            line_map[labels == i] = 0

    return line_map, enhanced


def _get_ordered_contour_points(mask):
    """Extract the ordered point list of the longest contour in a mask."""
    if mask.max() == 0:
        return np.array([])
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([])
    longest = max(contours, key=cv2.contourArea)
    return longest.reshape(-1, 2)


# ── PHASE 2: FINE LINE DETECTION ──────────────────────────────────────────

def detect_breaks(mask):
    """Detect breaks (gaps) in a line mask. Returns list of break positions (0-1 normalized)."""
    if mask.max() == 0:
        return []
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 1:
        return []
    # Multiple contour fragments = breaks exist
    # Sort by x-position to get ordered break locations
    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"], cv2.arcLength(c, False)))
    centroids.sort(key=lambda p: p[0])
    if len(centroids) < 2:
        return []
    h, w = mask.shape[:2]
    breaks = []
    for i in range(len(centroids) - 1):
        gap_x = (centroids[i][0] + centroids[i + 1][0]) / 2
        gap_y = (centroids[i][1] + centroids[i + 1][1]) / 2
        breaks.append({"position": round(gap_x / w, 3), "y_position": round(gap_y / h, 3)})
    return breaks


def detect_branches(mask, enhanced_line_map):
    """Detect branches extending from a major line using skeleton junction analysis."""
    if mask.max() == 0:
        return {"upward": 0, "downward": 0, "total": 0}
    # Dilate the mask slightly to find connected fine lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    # Find fine lines that connect to this major line
    connected_fines = cv2.bitwise_and(enhanced_line_map, dilated)
    # Remove the original line itself
    branch_only = cv2.subtract(connected_fines, mask)
    if branch_only.max() == 0:
        return {"upward": 0, "downward": 0, "total": 0}
    # Count distinct branch segments
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(branch_only, connectivity=8)
    min_branch_area = 8
    # Classify branches as upward or downward relative to the main line centroid
    main_pts = _get_ordered_contour_points(mask)
    if len(main_pts) == 0:
        return {"upward": 0, "downward": 0, "total": 0}
    main_center_y = np.mean(main_pts[:, 1])
    upward = 0
    downward = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_branch_area:
            branch_cy = centroids[i][1]
            if branch_cy < main_center_y:
                upward += 1
            else:
                downward += 1
    return {"upward": upward, "downward": downward, "total": upward + downward}


def detect_islands(mask):
    """Detect island formations (enclosed oval shapes) on a line."""
    if mask.max() == 0:
        return 0
    binary = (mask > 0).astype(np.uint8) * 255
    # Islands appear as enclosed loops — look for inner contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    island_count = 0
    for i, h in enumerate(hierarchy[0]):
        # h = [next, prev, child, parent]
        if h[3] != -1:  # has parent = inner contour = potential island
            area = cv2.contourArea(contours[i])
            if 5 < area < 500:
                island_count += 1
    return island_count


def detect_fork(mask):
    """Detect if a line ends in a fork. Returns fork type and position."""
    pts = _get_ordered_contour_points(mask)
    if len(pts) < 20:
        return {"has_fork": False, "type": "none"}
    h, w = mask.shape[:2]
    # Look at the last 20% of the line for branching
    end_region = pts[int(len(pts) * 0.8):]
    if len(end_region) < 5:
        return {"has_fork": False, "type": "none"}
    # Check for spread at the end by measuring width variation
    y_vals = end_region[:, 1]
    y_range = np.ptp(y_vals)
    # Compare with middle section spread
    mid_region = pts[int(len(pts) * 0.3):int(len(pts) * 0.6)]
    if len(mid_region) < 5:
        return {"has_fork": False, "type": "none"}
    mid_y_range = np.ptp(mid_region[:, 1])
    if mid_y_range > 0 and y_range / mid_y_range > 1.8:
        return {
            "has_fork": True,
            "type": "writers_fork" if y_range > 15 else "small_fork",
            "spread_ratio": round(y_range / max(mid_y_range, 1), 2),
        }
    return {"has_fork": False, "type": "none"}


def detect_chain_regions(mask, enhanced_gray):
    """Detect chained (wavy/oscillating) regions along a line from depth variance."""
    pts = _get_ordered_contour_points(mask)
    if len(pts) < 30:
        return {"chain_ratio": 0.0, "chain_segments": 0}
    h, w = mask.shape[:2]
    # Sample intensity along the line to detect oscillating depth
    n_samples = min(50, len(pts))
    indices = np.linspace(0, len(pts) - 1, n_samples, dtype=int)
    intensities = []
    for idx in indices:
        px, py = pts[idx]
        px, py = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        # Average intensity in 3x3 neighborhood
        y_lo, y_hi = max(0, py - 1), min(h, py + 2)
        x_lo, x_hi = max(0, px - 1), min(w, px + 2)
        region = enhanced_gray[y_lo:y_hi, x_lo:x_hi]
        intensities.append(float(np.mean(region)) if region.size > 0 else 128.0)
    intensities = np.array(intensities)
    if len(intensities) < 10:
        return {"chain_ratio": 0.0, "chain_segments": 0}
    # Compute local variance — high variance = chain pattern
    window = 5
    local_vars = []
    for i in range(window, len(intensities) - window):
        local = intensities[i - window:i + window]
        local_vars.append(np.var(local))
    if not local_vars:
        return {"chain_ratio": 0.0, "chain_segments": 0}
    local_vars = np.array(local_vars)
    chain_threshold = np.median(local_vars) * 2.5
    chain_mask = local_vars > chain_threshold
    chain_ratio = float(np.mean(chain_mask))
    # Count distinct chain segments
    chain_segments = 0
    in_chain = False
    for v in chain_mask:
        if v and not in_chain:
            chain_segments += 1
            in_chain = True
        elif not v:
            in_chain = False
    return {"chain_ratio": round(chain_ratio, 3), "chain_segments": chain_segments}


# ── PHASE 3: DEPTH PROFILING ──────────────────────────────────────────────

def analyze_line_depth_profile(mask, enhanced_gray, n_samples=10):
    """Sample the line intensity at N evenly spaced points to reveal
    where the line is deep vs faint — directly maps to energy levels
    at different life periods. Also detects double/sister lines."""
    pts = _get_ordered_contour_points(mask)
    if len(pts) < n_samples:
        return {
            "depth_samples": [0.5] * n_samples,
            "depth_variance": 0.0,
            "avg_depth": 0.5,
            "deepest_region": 0.5,
            "faintest_region": 0.5,
            "has_sister_line": False,
        }
    h, w = mask.shape[:2]
    indices = np.linspace(0, len(pts) - 1, n_samples, dtype=int)
    depths = []
    for idx in indices:
        px, py = pts[idx]
        px, py = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        # Sample intensity in 5x5 neighborhood
        y_lo, y_hi = max(0, py - 2), min(h, py + 3)
        x_lo, x_hi = max(0, px - 2), min(w, px + 3)
        region = enhanced_gray[y_lo:y_hi, x_lo:x_hi]
        val = float(np.mean(region)) / 255.0 if region.size > 0 else 0.5
        depths.append(round(val, 3))
    # Invert so darker = deeper line (lower intensity = deeper crease)
    depths = [round(1.0 - d, 3) for d in depths]
    depth_variance = float(np.var(depths))
    avg_depth = float(np.mean(depths))
    deepest_idx = int(np.argmax(depths))
    faintest_idx = int(np.argmin(depths))

    # Detect sister/double line — check for parallel line nearby
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    border_region = cv2.subtract(dilated, cv2.dilate(mask, kernel, iterations=1))
    # If there's significant line presence in the border region it's a sister line
    border_line = cv2.bitwise_and(
        cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5),
        border_region,
    )
    sister_pixels = cv2.countNonZero(border_line)
    main_pixels = cv2.countNonZero(mask)
    has_sister = sister_pixels > main_pixels * 0.15

    return {
        "depth_samples": depths,
        "depth_variance": round(depth_variance, 4),
        "avg_depth": round(avg_depth, 3),
        "deepest_region": round(deepest_idx / max(n_samples - 1, 1), 3),
        "faintest_region": round(faintest_idx / max(n_samples - 1, 1), 3),
        "has_sister_line": has_sister,
    }


# ── PHASE 4: SPATIAL POSITION ANALYSIS ───────────────────────────────────

def compute_spatial_features(mask, label):
    """Extract precise spatial start/end positions and multi-point curvature
    for a major line. These are unique fingerprints per palm."""
    pts = _get_ordered_contour_points(mask)
    h, w = mask.shape[:2]
    if len(pts) < 5:
        return {
            f"{label}_start_x": 0.5, f"{label}_start_y": 0.5,
            f"{label}_end_x": 0.5, f"{label}_end_y": 0.5,
            f"{label}_span_x": 0.0, f"{label}_span_y": 0.0,
            f"{label}_curvature_samples": [1.0] * 5,
            f"{label}_bbox_area_ratio": 0.0,
        }
    # Sort by x to get left-to-right ordering
    sorted_pts = pts[pts[:, 0].argsort()]
    start = sorted_pts[0]
    end = sorted_pts[-1]

    # Multi-point curvature (sample at 5 segments)
    curvature_samples = []
    n_segments = 5
    seg_size = max(len(sorted_pts) // n_segments, 3)
    for i in range(n_segments):
        seg = sorted_pts[i * seg_size : min((i + 1) * seg_size, len(sorted_pts))]
        if len(seg) < 3:
            curvature_samples.append(1.0)
            continue
        arc = cv2.arcLength(seg.reshape(-1, 1, 2), closed=False)
        dist = np.linalg.norm(seg[0].astype(float) - seg[-1].astype(float))
        curvature_samples.append(round(arc / max(dist, 1.0), 3))

    # Bounding box area ratio (how much of the bbox the line fills)
    x_min, y_min = sorted_pts.min(axis=0)
    x_max, y_max = sorted_pts.max(axis=0)
    bbox_area = max((x_max - x_min) * (y_max - y_min), 1)
    line_pixel_count = cv2.countNonZero(mask)

    return {
        f"{label}_start_x": round(float(start[0]) / w, 3),
        f"{label}_start_y": round(float(start[1]) / h, 3),
        f"{label}_end_x": round(float(end[0]) / w, 3),
        f"{label}_end_y": round(float(end[1]) / h, 3),
        f"{label}_span_x": round(float(end[0] - start[0]) / w, 3),
        f"{label}_span_y": round(float(end[1] - start[1]) / h, 3),
        f"{label}_curvature_samples": curvature_samples,
        f"{label}_bbox_area_ratio": round(line_pixel_count / bbox_area, 4),
    }


def compute_line_gap(mask1, mask2):
    """Measure the minimum gap between two lines at their origin (near the thumb).
    This is the Life-Head gap that indicates independence timing."""
    pts1 = _get_ordered_contour_points(mask1)
    pts2 = _get_ordered_contour_points(mask2)
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0
    h, w = mask1.shape[:2]
    # Focus on the leftmost 30% (origin area near thumb)
    x_threshold = w * 0.3
    pts1_origin = pts1[pts1[:, 0] < x_threshold]
    pts2_origin = pts2[pts2[:, 0] < x_threshold]
    if len(pts1_origin) == 0 or len(pts2_origin) == 0:
        return 0.0
    # Find minimum distance between the two sets of points
    min_dist = float('inf')
    # Subsample for efficiency
    step1 = max(1, len(pts1_origin) // 30)
    step2 = max(1, len(pts2_origin) // 30)
    for p1 in pts1_origin[::step1]:
        for p2 in pts2_origin[::step2]:
            d = np.linalg.norm(p1.astype(float) - p2.astype(float))
            if d < min_dist:
                min_dist = d
    return round(min_dist / h, 4)


def detect_minor_lines(enhanced_line_map, major_mask_combined, mask_shape):
    """Detect minor/fine lines that are NOT part of the 3 major lines.
    These include fate, sun, marriage, travel, children lines etc.
    Uses lower thresholds to capture even faint, short creases."""
    # Dilate major lines slightly so we don't double-count border pixels
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    major_dilated = cv2.dilate(major_mask_combined, kernel_dilate, iterations=1)
    # Remove major lines from the enhanced map
    fine_only = cv2.subtract(enhanced_line_map, major_dilated)
    if fine_only.max() == 0:
        return {"total_fine_lines": 0, "fine_line_density": 0.0, "fine_line_total_length": 0.0,
                "fine_line_lengths": [], "longest_fine_line": 0.0}
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fine_only, connectivity=8)
    min_line_area = 5      # Very low — capture even tiny faint lines
    min_arc_length = 8     # Lower threshold — keep short creases
    valid_lines = 0
    total_fine_length = 0.0
    fine_line_lengths = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_line_area:
            # Compute arc length for this component
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                arc = cv2.arcLength(max(contours, key=cv2.contourArea), False)
                if arc > min_arc_length:
                    valid_lines += 1
                    total_fine_length += arc
                    fine_line_lengths.append(round(arc, 1))
    h, w = mask_shape[:2]
    density = total_fine_length / (h * w) * 10000  # per 10k pixels
    fine_line_lengths.sort(reverse=True)
    return {
        "total_fine_lines": valid_lines,
        "fine_line_density": round(density, 2),
        "fine_line_total_length": round(total_fine_length, 1),
        "fine_line_lengths": fine_line_lengths[:20],  # Top 20 longest
        "longest_fine_line": fine_line_lengths[0] if fine_line_lengths else 0.0,
    }


# ── PHASE 5: MASTER FEATURE EXTRACTION (60+ features) ────────────────────

def get_line_length(mask):
    if mask.max() == 0: return 0
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return 0
    longest_contour = max(contours, key=cv2.contourArea)
    return cv2.arcLength(longest_contour, closed=False)

def get_curvature(mask):
    if mask.max() == 0: return 0
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return 0
    longest_contour = max(contours, key=cv2.contourArea)
    arc_length = cv2.arcLength(longest_contour, closed=False)
    contour_points = longest_contour.reshape(-1, 2)
    if len(contour_points) < 2: return 0
    straight_distance = np.linalg.norm(contour_points[0] - contour_points[-1])
    if straight_distance < 1: return 1.0
    return arc_length / straight_distance

def get_line_angle(mask):
    if mask.max() == 0: return 0
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return 0
    longest_contour = max(contours, key=cv2.contourArea)
    [vx, vy, x, y] = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    return float(angle[0]) if isinstance(angle, np.ndarray) else float(angle)

def count_intersections(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    if intersection.max() == 0: return 0
    num_labels, _ = cv2.connectedComponents(intersection)
    return max(0, num_labels - 1)


def extract_palm_features(segmentation_mask, raw_image=None):
    """Extract 60+ unique features from the palm using both the CNN segmentation
    mask AND raw image analysis. When raw_image is None, falls back to basic mode."""
    life_mask = (segmentation_mask == 1).astype(np.uint8) * 255
    head_mask = (segmentation_mask == 2).astype(np.uint8) * 255
    heart_mask = (segmentation_mask == 3).astype(np.uint8) * 255

    features = {}

    if raw_image is None:
        features['life_length'] = get_line_length(life_mask)
        features['head_length'] = get_line_length(head_mask)
        features['heart_length'] = get_line_length(heart_mask)
        features['life_curvature'] = get_curvature(life_mask)
        features['head_curvature'] = get_curvature(head_mask)
        features['heart_curvature'] = get_curvature(heart_mask)
        features['life_angle'] = get_line_angle(life_mask)
        features['head_angle'] = get_line_angle(head_mask)
        features['heart_angle'] = get_line_angle(heart_mask)
        features['life_head_intersection'] = count_intersections(life_mask, head_mask)
        features['life_heart_intersection'] = count_intersections(life_mask, heart_mask)
        features['head_heart_intersection'] = count_intersections(head_mask, heart_mask)
        return features

    # ══════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURES — derived from raw image analysis
    # ══════════════════════════════════════════════════════════════════════
    enhanced_line_map, enhanced_gray = enhance_palm_image(raw_image)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined_life_mask = cv2.bitwise_and(enhanced_line_map, cv2.dilate(life_mask, kernel_dilate))
    refined_head_mask = cv2.bitwise_and(enhanced_line_map, cv2.dilate(head_mask, kernel_dilate))
    refined_heart_mask = cv2.bitwise_and(enhanced_line_map, cv2.dilate(heart_mask, kernel_dilate))

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_life = cv2.morphologyEx(refined_life_mask, cv2.MORPH_CLOSE, kernel_close)
    closed_head = cv2.morphologyEx(refined_head_mask, cv2.MORPH_CLOSE, kernel_close)
    closed_heart = cv2.morphologyEx(refined_heart_mask, cv2.MORPH_CLOSE, kernel_close)

    # ── Basic features using highly-variable exact edges ──
    features['life_length'] = get_line_length(closed_life) or get_line_length(life_mask)
    features['head_length'] = get_line_length(closed_head) or get_line_length(head_mask)
    features['heart_length'] = get_line_length(closed_heart) or get_line_length(heart_mask)
    features['life_curvature'] = get_curvature(closed_life) or get_curvature(life_mask)
    features['head_curvature'] = get_curvature(closed_head) or get_curvature(head_mask)
    features['heart_curvature'] = get_curvature(closed_heart) or get_curvature(heart_mask)
    features['life_angle'] = get_line_angle(closed_life) or get_line_angle(life_mask)
    features['head_angle'] = get_line_angle(closed_head) or get_line_angle(head_mask)
    features['heart_angle'] = get_line_angle(closed_heart) or get_line_angle(heart_mask)
    features['life_head_intersection'] = count_intersections(closed_life, closed_head)
    features['life_heart_intersection'] = count_intersections(closed_life, closed_heart)
    features['head_heart_intersection'] = count_intersections(closed_head, closed_heart)

    masks = {"life": closed_life, "head": closed_head, "heart": closed_heart}

    for name, mask in masks.items():
        # Breaks
        breaks = detect_breaks(mask)
        features[f'{name}_break_count'] = len(breaks)
        features[f'{name}_breaks'] = breaks

        # Branches
        branches = detect_branches(mask, enhanced_line_map)
        features[f'{name}_branch_up'] = branches["upward"]
        features[f'{name}_branch_down'] = branches["downward"]
        features[f'{name}_branch_total'] = branches["total"]

        # Islands
        features[f'{name}_island_count'] = detect_islands(mask)

        # Fork at end
        fork = detect_fork(mask)
        features[f'{name}_has_fork'] = fork["has_fork"]
        features[f'{name}_fork_type'] = fork["type"]

        # Chain detection
        chains = detect_chain_regions(mask, enhanced_gray)
        features[f'{name}_chain_ratio'] = chains["chain_ratio"]
        features[f'{name}_chain_segments'] = chains["chain_segments"]

        # Depth profiling (10 samples along the line)
        depth = analyze_line_depth_profile(mask, enhanced_gray, n_samples=10)
        features[f'{name}_depth_samples'] = depth["depth_samples"]
        features[f'{name}_depth_variance'] = depth["depth_variance"]
        features[f'{name}_avg_depth'] = depth["avg_depth"]
        features[f'{name}_deepest_region'] = depth["deepest_region"]
        features[f'{name}_faintest_region'] = depth["faintest_region"]
        features[f'{name}_has_sister_line'] = depth["has_sister_line"]

        # Spatial positions
        spatial = compute_spatial_features(mask, name)
        features.update(spatial)

    # ── Cross-line spatial relationships ──
    features['life_head_gap'] = compute_line_gap(life_mask, head_mask)
    features['head_heart_gap'] = compute_line_gap(head_mask, heart_mask)

    # ── Minor/fine line detection ──
    major_combined = cv2.bitwise_or(cv2.bitwise_or(life_mask, head_mask), heart_mask)
    minor_lines = detect_minor_lines(enhanced_line_map, major_combined, segmentation_mask.shape)
    features.update(minor_lines)

    # ── Unique palm signature (hash of all continuous features for dedup) ──
    sig_vals = [
        features['life_length'], features['head_length'], features['heart_length'],
        features['life_curvature'], features['head_curvature'], features['heart_curvature'],
        features['life_angle'], features['head_angle'], features['heart_angle'],
        features.get('life_avg_depth', 0), features.get('head_avg_depth', 0),
        features.get('heart_avg_depth', 0),
        features.get('life_branch_total', 0), features.get('head_branch_total', 0),
        features.get('life_break_count', 0), features.get('total_fine_lines', 0),
    ]
    import hashlib
    sig_str = "_".join([f"{v:.3f}" if isinstance(v, float) else str(v) for v in sig_vals])
    features['palm_signature'] = hashlib.md5(sig_str.encode()).hexdigest()[:12]

    return features


def classify_palm(features):
    classification = {}
    lengths = {'Life': features.get('life_length', 0), 'Head': features.get('head_length', 0), 'Heart': features.get('heart_length', 0)}
    total_length = sum(lengths.values())
    if total_length == 0:
        classification['dominant_line'] = 'Unknown'
        classification['confidence'] = 0.0
    else:
        dominant_line = max(lengths, key=lengths.get)
        classification['dominant_line'] = dominant_line
        classification['confidence'] = round(lengths[dominant_line] / total_length, 3)

    avg_curvature = (features.get('life_curvature', 0) + features.get('head_curvature', 0) + features.get('heart_curvature', 0)) / 3
    if avg_curvature > 1.3: classification['palm_type'] = 'Curved/Expressive'
    elif avg_curvature > 1.1: classification['palm_type'] = 'Balanced'
    else: classification['palm_type'] = 'Straight/Practical'

    head_angle = abs(features.get('head_angle', 0))
    intersections = features.get('life_head_intersection', 0)
    if head_angle > 10 and intersections > 0:
        classification['career_shift_indicator'] = 'Yes'
        classification['career_shift_confidence'] = 0.7
    else:
        classification['career_shift_indicator'] = 'No'
        classification['career_shift_confidence'] = 0.6
    return classification


def create_palm_overlay(image, mask, enhanced_line_map=None):
    """Create overlay showing major lines (colored) plus fine lines (cyan)."""
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    # Major lines in vivid colors (Life=Red, Head=Green, Heart=Blue)
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
    
    if enhanced_line_map is not None:
        if enhanced_line_map.shape[:2] != image.shape[:2]:
            enhanced_line_map = cv2.resize(enhanced_line_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Draw highly precise major lines
        kernel_dilate = np.ones((7,7), np.uint8)
        kernel_thicken = np.ones((3,3), np.uint8)
        major_combined = np.zeros_like(mask, dtype=np.uint8)
        
        for class_id, color in colors.items():
            class_mask = (mask == class_id).astype(np.uint8)
            dilated_mask = cv2.dilate(class_mask, kernel_dilate, iterations=1)
            major_combined = cv2.bitwise_or(major_combined, dilated_mask)
            
            actual_line = cv2.bitwise_and(enhanced_line_map, enhanced_line_map, mask=dilated_mask)
            thick_line = cv2.dilate(actual_line, kernel_thicken, iterations=1)
            
            overlay_mask = thick_line > 0
            overlay[overlay_mask] = overlay[overlay_mask] * 0.2 + np.array(color) * 0.8
            
        # Fine lines
        fine_only = cv2.subtract(enhanced_line_map, major_combined)
        fine_mask = fine_only > 0
        overlay[fine_mask] = overlay[fine_mask] * 0.4 + np.array([255, 255, 0]) * 0.6  # Cyan/yellow tint
    else:
        for class_id, color in colors.items():
            class_mask = (mask == class_id)
            overlay[class_mask] = overlay[class_mask] * 0.5 + np.array(color) * 0.5
            
    return overlay.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PALM OBSERVATIONS — derived from actual detected features per palm
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OBSERVATIONS = {
    "dominant_hand": "Right",
    "hand_shape": "Auto / unsure",
    "line_depth": "Medium",
    "major_breaks": "A few",
    "fate_line": "Faint",
    "sun_line": "Faint",
}


def derive_dynamic_observations(features):
    """Derive palmistry observations dynamically from actual CV-detected features.
    Each palm scan produces unique, accurate observations instead of relying
    on hardcoded defaults. This ensures the analysis updates per-palm."""
    obs = {}

    # Dominant hand — cannot be detected from image, default to Right
    obs["dominant_hand"] = "Right"

    # Hand shape — auto-detect from features (the engine will classify)
    obs["hand_shape"] = "Auto / unsure"

    # Line depth — derive from the averaged depth across all 3 major lines
    avg_depths = []
    for key in ('life', 'head', 'heart'):
        d = features.get(f'{key}_avg_depth', None)
        if d is not None:
            avg_depths.append(float(d))
    if avg_depths:
        overall_depth = sum(avg_depths) / len(avg_depths)
        if overall_depth > 0.55:
            obs["line_depth"] = "Deep"
        elif overall_depth > 0.30:
            obs["line_depth"] = "Medium"
        else:
            obs["line_depth"] = "Faint"
    else:
        obs["line_depth"] = "Medium"

    # Major breaks — derive from actual break counts detected by CV
    total_breaks = sum(int(features.get(f'{k}_break_count', 0)) for k in ('life', 'head', 'heart'))
    if total_breaks == 0:
        obs["major_breaks"] = "None"
    elif total_breaks <= 3:
        obs["major_breaks"] = "A few"
    else:
        obs["major_breaks"] = "Many"

    # Fate line visibility — infer from minor line analysis
    total_fine = int(features.get('total_fine_lines', 0))
    fine_density = float(features.get('fine_line_density', 0))
    longest_fine = float(features.get('longest_fine_line', 0))

    if longest_fine > 80 and fine_density > 4.0:
        obs["fate_line"] = "Strong"
    elif longest_fine > 40 and fine_density > 2.0:
        obs["fate_line"] = "Clear"
    elif total_fine > 2:
        obs["fate_line"] = "Faint"
    else:
        obs["fate_line"] = "Not visible"

    # Sun line visibility — similar inference from fine line characteristics
    if longest_fine > 60 and total_fine > 8:
        obs["sun_line"] = "Clear"
    elif total_fine > 4:
        obs["sun_line"] = "Faint"
    else:
        obs["sun_line"] = "Not visible"

    return obs


def _draw_live_palm_summary(image, report):
    output = image.copy()
    labels = [
        f"Dominant: {report['dominant_line']}",
        f"Quality: {report['detection_quality']:.2f}",
        f"Career Shift: {report['career_shift_indicator']}",
    ]
    if report["detection_quality"] < 0.55:
        labels.append("Tip: move closer, flatten palm, use brighter light")

    panel_height = 34 + len(labels) * 22
    cv2.rectangle(output, (10, 10), (470, panel_height), (5, 10, 24), thickness=-1)
    for idx, text in enumerate(labels):
        cv2.putText(
            output,
            text,
            (20, 36 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (236, 245, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def _render_palm_report(overlay, features, report):
    from utils.voice import render_voice_button
    import uuid

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — PALM SCAN (full-width image + key metrics)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin:10px 0 20px;">
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
            <span style="font-size:14px;color:#F59E0B;letter-spacing:4px;
                font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🖐️ Palm Scan Results</span>
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption="Line overlay — Major: Life (red) · Head (green) · Heart (blue) | Fine lines: cyan",
        )
    with c2:
        dq = report.get("detection_quality", 1.0)
        ht = report.get("hand_type", {})
        personality = report.get("personality", {})
        if dq < 0.55:
            render_content_card(
                "⚠️ Scan Quality Low",
                "Better lighting and a flatter palm will improve the reading.",
                accent_color="#F59E0B", icon="⚠️",
            )
        fine_detail = features.get('total_fine_lines', 0)
        render_info_grid([
            ("Hand Type", f"{ht.get('type', 'Mixed')}"),
            ("Element", ht.get('element', 'Mixed')),
            ("Dominant Line", report["dominant_line"]),
            ("Detection", f"{dq:.0%}"),
            ("Dominant Mount", report.get('dominant_mount', 'Unknown').replace('_', ' ')),
            ("Fine Lines", str(fine_detail)),
            ("Archetype", personality.get('archetype', 'Unknown')),
            ("Palm ID", features.get('palm_signature', 'N/A')),
        ])

    # Summary card — full width
    render_content_card(
        "🔮 Professional Reading Summary",
        report["summary"].replace('\n', '<br>'),
        accent_color="#8B5CF6", icon="🔮",
    )
    render_voice_button(report["summary"], key_suffix=f"palm_summary_{uuid.uuid4().hex[:8]}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2 & 3 — EXPLORATION + INTERPRETATION (tabs)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin:30px 0 20px;">
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
            <span style="font-size:14px;color:#F59E0B;letter-spacing:4px;
                font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🔍 Deep Analysis</span>
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        </div>
    """, unsafe_allow_html=True)

    line_strength_df = pd.DataFrame(
        {
            "line": list(report["line_strengths"].keys()),
            "strength": [round(v * 100, 1) for v in report["line_strengths"].values()],
        }
    ).set_index("line")

    # Tabbed phases
    tab_lines, tab_fine, tab_mounts, tab_timing, tab_personality, tab_health, tab_features, tab_raw = st.tabs([
        "📜 Line Interpretation",
        "🔬 Fine Line Detail",
        "⛰️ Mount Analysis",
        "⏳ Time Predictions",
        "👤 Personality",
        "💚 Health",
        "📊 Feature Dashboard",
        "🗂️ Raw Data",
    ])

    # ── TAB 1: LINE INTERPRETATION ──
    with tab_lines:
        line_icons = {"Life": "❤️", "Head": "🧠", "Heart": "💙"}
        line_colors = {"Life": "#10B981", "Head": "#06B6D4", "Heart": "#EC4899"}
        for item in report["line_readings"]:
            governs_html = "".join([f"<span style='background:rgba(255,255,255,0.05);padding:2px 8px;border-radius:12px;font-size:11px;color:#94A3B8;margin:2px;display:inline-block;'>{g}</span>" for g in item.get('governs', [])[:4]])
            render_content_card(
                f"{item['line']} Line ({item.get('hindi', '')})",
                f"{item['detail']}<br><br>"
                f"<div style='margin-top:8px;'>{governs_html}</div><br>"
                f"<span style='color:#64748B;font-size:11px;'>Depth: {item.get('depth', 'Medium')} · Shape: {item.get('shape', '')} · Prominence: {item.get('prominence', '')}</span>",
                accent_color=line_colors.get(item["line"], "#3B82F6"),
                icon=line_icons.get(item["line"], "〰️"),
            )

        st.markdown("#### Interpretation Themes")
        theme_cards = [
            ("🧠", "Mindset", "mindset", "#06B6D4"),
            ("❤️", "Relationships", "relationships", "#EC4899"),
            ("⚡", "Energy & Vitality", "energy", "#F59E0B"),
            ("💼", "Career & Fate", "career", "#10B981"),
            ("✨", "Fame & Visibility", "visibility", "#8B5CF6"),
            ("🔄", "Life Stability", "stability", "#3B82F6"),
            ("✋", "Hand Dominance", "dominant_hand", "#F59E0B"),
            ("📏", "Line Depth", "line_depth", "#06B6D4"),
        ]
        tc1, tc2 = st.columns(2)
        for idx, (icon, title, key, color) in enumerate(theme_cards):
            with tc1 if idx % 2 == 0 else tc2:
                render_content_card(title, report["themes"].get(key, ""), accent_color=color, icon=icon)

        notes_html = "".join([f"<div style='margin-bottom:6px;'>• {note}</div>" for note in report["shared_notes"]])
        render_content_card("Pattern Notes", notes_html, accent_color="#8B5CF6", icon="📝")

    # ── TAB 2: MOUNT ANALYSIS ──
    with tab_mounts:
        st.markdown("#### Mount Prominence Analysis")
        st.caption("Mounts are the fleshy pads on the palm. Each is governed by a planet and reveals personality, career, and relationship tendencies.")
        mounts = report.get("mounts", {})
        mount_colors = {"Jupiter": "#F59E0B", "Saturn": "#6366F1", "Sun_Apollo": "#EAB308", "Mercury": "#06B6D4", "Venus": "#EC4899", "Moon": "#8B5CF6", "Mars": "#EF4444"}
        mc1, mc2 = st.columns(2)
        for idx, (name, data) in enumerate(mounts.items()):
            strength = data.get('strength', 'Unknown')
            score = data.get('score', 0)
            strength_color = '#10B981' if 'Well' in strength else '#F59E0B' if 'Over' in strength else '#64748B'
            reading = data.get('reading', {})
            personality_text = ""
            if isinstance(reading, dict):
                p = reading.get('personality', '')
                if isinstance(p, list):
                    personality_text = '<br>'.join([f"• {t}" for t in p[:3]])
                else:
                    personality_text = str(p)
            else:
                personality_text = str(reading)
            bar_width = int(score * 100)
            mount_html = (
                f"<div style='margin-bottom:4px;'>"
                f"<span style='color:{strength_color};font-weight:700;'>{strength}</span> "
                f"<span style='color:#64748B;'>(Score: {score})</span></div>"
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;height:8px;margin:8px 0;'>"
                f"<div style='background:{mount_colors.get(name, '#3B82F6')};height:100%;border-radius:8px;width:{bar_width}%;'></div></div>"
                f"<div style='font-size:12px;color:#CBD5E1;line-height:1.6;'>{personality_text[:300]}</div>"
            )
            with mc1 if idx % 2 == 0 else mc2:
                render_content_card(f"{name.replace('_', ' ')} Mount", mount_html, accent_color=mount_colors.get(name, "#3B82F6"), icon="⛰️")

    # ── TAB: FINE LINE DETAIL ──
    with tab_fine:
        st.markdown("#### Fine Line Analysis")
        st.caption("Advanced computer vision detects branches, breaks, islands, chains, and depth variations unique to your palm.")
        line_names = ["Life", "Head", "Heart"]
        line_keys = ["life", "head", "heart"]
        fine_colors = {"life": "#10B981", "head": "#06B6D4", "heart": "#EC4899"}
        fine_icons = {"life": "❤️", "head": "🧠", "heart": "💙"}
        for lname, lkey in zip(line_names, line_keys):
            detail_parts = []
            # Breaks
            bc = features.get(f'{lkey}_break_count', 0)
            if bc > 0:
                detail_parts.append(f"<b>Breaks:</b> {bc} break(s) detected — indicates transition points or life changes")
            else:
                detail_parts.append("<b>Breaks:</b> None — continuous, steady flow")
            # Branches
            bu = features.get(f'{lkey}_branch_up', 0)
            bd = features.get(f'{lkey}_branch_down', 0)
            if bu + bd > 0:
                detail_parts.append(f"<b>Branches:</b> {bu} upward (success/elevation) · {bd} downward (challenges/effort)")
            else:
                detail_parts.append("<b>Branches:</b> No visible branches")
            # Islands
            ic = features.get(f'{lkey}_island_count', 0)
            if ic > 0:
                detail_parts.append(f"<b>Islands:</b> {ic} — periods of uncertainty or health sensitivity")
            # Fork
            ft = features.get(f'{lkey}_fork_type', 'none')
            if ft != 'none':
                detail_parts.append(f"<b>Fork:</b> {ft.replace('_', ' ').title()} fork detected at line end")
            # Chain regions
            cr = features.get(f'{lkey}_chain_ratio', 0)
            if cr > 0.05:
                detail_parts.append(f"<b>Chains:</b> {cr:.0%} of line shows chain pattern — variable intensity periods")
            # Sister line
            if features.get(f'{lkey}_has_sister_line', False):
                detail_parts.append("<b>Sister Line:</b> Parallel protection line detected — extra vitality/support")
            # Depth profile
            depth_samples = features.get(f'{lkey}_depth_samples', [])
            if depth_samples:
                avg_d = features.get(f'{lkey}_avg_depth', 0.5)
                depth_label = 'Deep' if avg_d > 0.6 else 'Faint' if avg_d < 0.35 else 'Medium'
                detail_parts.append(f"<b>Avg Depth:</b> {depth_label} ({avg_d:.2f}) · Variance: {features.get(f'{lkey}_depth_variance', 0):.4f}")
            html = '<br>'.join(detail_parts)
            render_content_card(
                f"{lname} Line — Fine Detail",
                html,
                accent_color=fine_colors.get(lkey, "#3B82F6"),
                icon=fine_icons.get(lkey, "〰️"),
            )
        # Minor lines summary with dynamic observations
        total_fine = features.get('total_fine_lines', 0)
        fine_density = features.get('fine_line_density', 0)
        gap_info = features.get('life_head_gap', 0)
        longest_fine = features.get('longest_fine_line', 0)
        fine_lengths = features.get('fine_line_lengths', [])
        # Show dynamic observation values derived from THIS palm
        obs = report.get('observations', {})
        minor_html = (
            f"<b>Total minor/fine lines:</b> {total_fine}<br>"
            f"<b>Fine line density:</b> {fine_density:.1f} per 10k px<br>"
            f"<b>Longest fine line:</b> {longest_fine:.1f} px<br>"
            f"<b>Top fine line lengths:</b> {', '.join(str(l) for l in fine_lengths[:8])} px<br>"
            f"<b>Life-Head origin gap:</b> {gap_info:.4f} (larger = earlier independence)<br>"
            f"<hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:8px 0;'>"
            f"<b>🔄 Dynamic Observations (per-palm):</b><br>"
            f"<b>Line Depth:</b> {obs.get('line_depth', 'N/A')} | "
            f"<b>Breaks:</b> {obs.get('major_breaks', 'N/A')} | "
            f"<b>Fate Line:</b> {obs.get('fate_line', 'N/A')} | "
            f"<b>Sun Line:</b> {obs.get('sun_line', 'N/A')}<br>"
            f"<b>Palm Signature:</b> <code>{features.get('palm_signature', 'N/A')}</code>"
        )
        render_content_card("Minor Lines & Palm Fingerprint", minor_html, accent_color="#F59E0B", icon="🔍")

    # ── TAB 3: TIME PREDICTIONS ──
    with tab_timing:
        st.markdown("#### Time Predictions")
        st.caption("Based on proportional timing analysis applied to your detected line positions.")
        timing = report.get("timing", {})
        cat_icons = {"life_transition": "🔄", "career": "💼", "health_energy": "⚡", "relationships": "💕", "spiritual": "🔮"}
        cat_colors = {"life_transition": "#3B82F6", "career": "#10B981", "health_energy": "#F59E0B", "relationships": "#EC4899", "spiritual": "#8B5CF6"}
        for pred in timing.get("predictions", []):
            cat = pred.get("category", "life_transition")
            render_content_card(
                f"{pred.get('period', '')} — {pred.get('event', '')}",
                pred.get('detail', ''),
                accent_color=cat_colors.get(cat, "#3B82F6"),
                icon=cat_icons.get(cat, "📅"),
            )
        render_content_card("⚠️ Note", timing.get("note", ""), accent_color="#64748B", icon="ℹ️")

    # ── TAB 4: PERSONALITY ──
    with tab_personality:
        st.markdown("#### Personality Profile")
        personality = report.get("personality", {})
        ht = report.get("hand_type", {})
        render_info_grid([
            ("Archetype", personality.get('archetype', 'Unknown')),
            ("Hand Type", f"{ht.get('type', 'Mixed')} ({ht.get('hindi', '')})"),
            ("Element", ht.get('element', 'Mixed')),
            ("Dominant Mount", personality.get('dominant_mount', 'Unknown')),
        ])
        render_content_card(
            "Archetype Description",
            personality.get('description', ''),
            accent_color="#8B5CF6", icon="👤",
        )
        traits_html = "".join([f"<div style='margin-bottom:6px;padding:6px 12px;background:rgba(255,255,255,0.03);border-radius:8px;border-left:3px solid #3B82F6;'>• {t}</div>" for t in personality.get('core_traits', [])[:8]])
        render_content_card("Core Character Traits", traits_html, accent_color="#06B6D4", icon="✨")
        render_content_card(
            "Hand Type Profile",
            ht.get('description', ''),
            accent_color="#F59E0B", icon="✋",
        )
        career_html = ", ".join(ht.get('career', [])[:8])
        render_content_card("Career Aptitude", career_html, accent_color="#10B981", icon="💼")
        render_content_card("Relationship Style", ht.get('relationships', ''), accent_color="#EC4899", icon="💕")

    # ── TAB 5: HEALTH ──
    with tab_health:
        st.markdown("#### Health & Vitality Assessment")
        health = report.get("health", {})
        vitality = health.get('overall_vitality', 'moderate').title()
        vitality_color = '#10B981' if 'Strong' in vitality else '#F59E0B' if 'Sensitive' in vitality else '#3B82F6'
        render_info_grid([("Overall Vitality", vitality)])
        for ind in health.get("indicators", []):
            assessment = ind.get('assessment', 'Unknown')
            a_color = '#10B981' if assessment in ('Strong', 'Stable', 'Balanced') else '#F59E0B'
            render_content_card(
                f"{ind.get('area', '')} — {assessment}",
                ind.get('detail', ''),
                accent_color=a_color, icon="💚",
            )
        render_content_card("Medical Disclaimer", health.get('disclaimer', ''), accent_color="#EF4444", icon="⚕️")

    # ── TAB 6: FEATURE DASHBOARD ──
    with tab_features:
        st.caption("Detected line balance")
        st.bar_chart(line_strength_df, height=260)
        st.caption("Palm-reading prompts — ask these in the chatbot below")
        for question in report["questions"]:
            st.markdown(f"- {question}")

        guidance_html = "".join([f"<div style='margin-bottom:6px;'>🧭 {item}</div>" for item in report["guidance"]])
        render_content_card("Guidance", guidance_html, accent_color="#06B6D4", icon="🧭")

    # ── TAB 7: RAW DATA ──
    with tab_raw:
        st.json(
            {
                "report": {
                    "dominant_line": report["dominant_line"],
                    "dominant_strength_pct": report["dominant_strength_pct"],
                    "detection_quality": report["detection_quality"],
                    "hand_type": report.get("hand_type", {}).get("type", "Unknown"),
                    "archetype": report.get("personality", {}).get("archetype", "Unknown"),
                    "observations": report["observations"],
                },
                "features": {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()},
                "timing_predictions": [{"period": p["period"], "event": p["event"]} for p in report.get("timing", {}).get("predictions", [])],
                "mount_scores": {k: v.get("score", 0) for k, v in report.get("mounts", {}).items()},
            }
        )

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PALM READING (CNN Segmentation)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_palm_model(device):
    """Shared global cache for the Palm Segmentation model."""
    try:
        import segmentation_models_pytorch as smp
        import torch
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4)
        model.load_state_dict(torch.load("palm_model.pth", map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        st.error(f"Failed to load Palm Model from 'palm_model.pth': {exc}")
        return None

def _palm_module():
    section_header(
        "Professional Palm Analyzer",
        "Advanced CV palm analysis with CLAHE + Gabor filter bank · 60+ unique features · Fine line detection · Depth profiling",
    )
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(245,158,11,0.1));
                    border: 1px solid rgba(139,92,246,0.2); border-radius: 12px; padding: 16px 20px;
                    margin-bottom: 20px;">
            <span style="font-size: 16px;">🖐️</span>
            <span style="color: #E2E8F0; font-family: 'Inter', sans-serif; font-size: 14px;">
                Upload or capture your palm for a <strong>precision analysis</strong> covering
                <strong>personality, career, love, health, timing</strong> — powered by
                advanced computer vision that detects even the finest lines invisible to the eye.
                Use Camera Snapshot if live WebRTC gives STUN trouble.
            </span>
        </div>
    """, unsafe_allow_html=True)
    src = st.radio("Input Source", ["📷 Photo Upload", "📸 Camera Auto-Scan"], horizontal=True, key="cv_palm_src")
    observations = DEFAULT_OBSERVATIONS

    try:
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        st.error("Missing dependencies. Please run `pip install -r requirements.txt` (needs torch, albumentations).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_palm_model(device)
    if model is None: return
    preprocessing = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    def _palm_cb(img):
        best_angle = 0
        best_pred_mask = None
        best_area = -1
        best_rgb_img = None
        
        # Test 4 angles for true auto-alignment
        for angle in [0, 90, 180, 270]:
            if angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = img.copy()

            img_rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            preprocessed_rot = preprocessing(image=img_rgb_rot)
            image_tensor_rot = preprocessed_rot["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                output_rot = model(image_tensor_rot)
                pred_mask_rot = output_rot.argmax(dim=1).squeeze(0).cpu().numpy()

            area = np.sum(pred_mask_rot > 0)
            if area > best_area:
                best_area = area
                best_angle = angle
                best_pred_mask = pred_mask_rot
                best_rgb_img = rotated

        # Use the precisely aligned image and mask
        img = best_rgb_img
        mask = cv2.resize(best_pred_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Advanced: enhance raw image and extract 60+ features
        enhanced_line_map, _ = enhance_palm_image(img)
        overlay_img = create_palm_overlay(img, mask, enhanced_line_map)
        features = extract_palm_features(mask, raw_image=img)
        # Dynamic observations — derived from THIS palm's actual features
        dynamic_obs = derive_dynamic_observations(features)
        report = build_palm_report(features, dynamic_obs)
        return overlay_img, mask, features, report

    if src in ("📷 Photo Upload", "📸 Camera Auto-Scan"):
        img = _load_image_from_source(
            src,
            "Upload Palm Photo",
            "palm_photo",
            "Capture Palm Photo",
            "palm_camera",
        )
        if img is not None:
            # Clear previous palm state — ensures fresh per-palm analysis
            for key in list(st.session_state.keys()):
                if key.startswith("palm_latest_"):
                    del st.session_state[key]

            with st.spinner("🔬 Auto-aligning palm (4-axis scan) & extracting 60+ features via CLAHE/Gabor fusion..."):
                overlay, mask, features, report = _palm_cb(img.copy())

            st.session_state["palm_latest_report"] = report
            st.session_state["palm_latest_features"] = features
            st.session_state["palm_latest_summary"] = report["summary"]
            _render_palm_report(overlay, features, report)

    latest_report = st.session_state.get("palm_latest_report")
    if latest_report:
        st.caption("Latest saved scan summary")
        st.write(latest_report["summary"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _cv_module_renderer(module_key):
    renderers = {
        "attendance": _attendance_module,
        "face_scan": _face_scan_module,
        "vehicle": _vehicle_module,
        "sign": _sign_module,
        "palm": _palm_module,
    }
    return renderers[module_key]


def _render_cv_shell(title, subtitle, icon):
    from utils.styles import inject_global_css

    inject_global_css()
    gradient_header(title, subtitle, icon)

    if not WEBRTC_READY:
        st.error("`streamlit-webrtc` is missing. Features requiring live camera will not function.")


def _render_cv_jump_bar(active_key=None):
    options = [("OpenCV Gallery", CV_GALLERY_PATH)] + [(f"{item['icon']} {item['title']}", item["path"]) for item in CV_MODULES]
    labels = [label for label, _ in options]
    route_map = {label: path for label, path in options}

    current_label = "OpenCV Gallery"
    if active_key is not None:
        current_label = next(f"{item['icon']} {item['title']}" for item in CV_MODULES if item["key"] == active_key)

    c1, c2 = st.columns([1, 2.2])
    with c1:
        if active_key is None:
            st.button("📷 Gallery Home", use_container_width=True, disabled=True, key="cv_gallery_home_disabled")
        else:
            if st.button("⬅ Back to Gallery", use_container_width=True, key=f"cv_back_{active_key}"):
                st.switch_page(CV_GALLERY_PATH)
    with c2:
        selected = st.selectbox(
            "Jump to OpenCV page",
            labels,
            index=labels.index(current_label),
            key=f"cv_jump_{active_key or 'gallery'}",
        )
        if selected != current_label:
            st.switch_page(route_map[selected])


def _render_cv_tutor(module_key, topic_label):
    from utils.chatbot import render_chatbot, push_tutor_insight

    insight_key = f"cv_insight_{module_key}"
    if insight_key not in st.session_state:
        from utils.ai_helper import get_ai_explanation

        prompt = (
            f"The user is exploring the {topic_label}. "
            "Give a short expert tip explaining how it works in real time and what to watch in the output."
        )
        ai_text = get_ai_explanation(prompt)
        if ai_text:
            push_tutor_insight(ai_text, f"Vision Analyst // {topic_label.title()} Tips")
            st.session_state[insight_key] = True

    if module_key == "palm" and st.session_state.get("palm_latest_report"):
        palm_report = st.session_state["palm_latest_report"]
        render_chatbot(
            "Professional palm analysis — hand type, lines, mounts, timing, health, personality",
            context_payload=palm_report.get("chat_context"),
            system_prompt=build_professional_system_prompt(),
            fallback_builder=lambda question: answer_palm_question(question, palm_report),
            greeting=get_professional_greeting(),
            theme=MODULE_THEMES["opencv"],
            tutor_label="PALM ANALYSIS EXPERT 🖐️",
            placeholder="Ask about career, love, timing, health, personality, fortune...",
        )
        return

    render_chatbot(
        topic_label,
        system_prompt=(
            "You are a sharp, direct computer vision engineer. You explain OpenCV concepts with precision, "
            "use technical vocabulary confidently, and always connect theory to what the user sees in the output."
        ),
        greeting=(
            f"👁️ Vision Analyst online. I'm here to help you understand what the {topic_label} is detecting "
            "and why. Ask me about the algorithms, the output, or how to improve results."
        ),
        theme=MODULE_THEMES["opencv"],
        tutor_label="VISION ANALYST 📷",
        placeholder="Ask about the detection results...",
    )


def _render_cv_gallery():
    _render_cv_shell("Optical Analytics Hub", "Face Identity · Live Motion · Structural Analysis", "👁️")
    _render_cv_jump_bar()

    st.info("Each OpenCV tool below now opens on its own dedicated page, so you can focus on one vision workflow at a time.")
    st.markdown('<h3 style="font-family: \'Montserrat\', sans-serif; color: white; font-weight: 700; margin-bottom: 25px; border-bottom: 2px solid #06B6D4; display: inline-block; padding-bottom: 10px;">Modules Gallery</h3>', unsafe_allow_html=True)

    for idx, item in enumerate(CV_MODULES):
        card_color = "#06B6D4" if idx % 2 == 0 else "#3B82F6"
        with st.container():
            e_col1, e_col2, e_col3 = st.columns([1.2, 3, 1])

            with e_col1:
                st.image(item["banner"], width="stretch")

            with e_col2:
                st.markdown(
                    f"""
                    <div style="padding: 5px 0;">
                        <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 20px; color: white; margin-bottom: 5px;">{item['title']} {item['icon']}</div>
                        <p style="color: #F8FAFC; font-size: 14px; line-height: 1.5; margin-bottom: 12px; font-weight: 500;">{item['gallery_subtitle']}. Optimized for real-time vision processing.</p>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            {"".join([f'<span style="background: rgba(255,255,255,0.05); padding: 3px 10px; border-radius: 4px; color: {card_color}; font-size: 11px; font-weight: 600; border: 1px solid rgba(255,255,255,0.1);">{feature}</span>' for feature in item['features']])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with e_col3:
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                if st.button(f"Open {item['title']}", key=f"launch_cv_page_{item['key']}", type="primary", use_container_width=True):
                    st.switch_page(item["path"])

            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;'>", unsafe_allow_html=True)

    from utils.chatbot import render_chatbot

    render_chatbot(
        "Computer Vision Analytics Hub",
        system_prompt=(
            "You are a sharp, direct computer vision engineer. You explain OpenCV concepts with precision "
            "and always connect theory to what the user sees in the output."
        ),
        greeting="👁️ Vision Analyst online. Select a module above to begin, or ask me anything about computer vision.",
        theme=MODULE_THEMES["opencv"],
        tutor_label="VISION ANALYST 📷",
        placeholder="Ask about computer vision...",
    )


def _render_cv_module_page(module_key):
    module = CV_MODULE_MAP[module_key]
    inject_module_theme("opencv")
    _render_cv_shell(module["page_title"], module["page_subtitle"], module["icon"])
    _render_cv_jump_bar(module_key)
    render_content_card(
        module["page_title"],
        module["page_subtitle"],
        accent_color="#F59E0B",
        icon=module["icon"],
    )
    _cv_module_renderer(module_key)()
    st.divider()
    _render_cv_tutor(module_key, f"{module['title']} computer vision module")


def opencv_detection_page():
    _render_cv_gallery()


def opencv_attendance_page():
    _render_cv_module_page("attendance")


def opencv_face_scan_page():
    _render_cv_module_page("face_scan")


def opencv_vehicle_page():
    _render_cv_module_page("vehicle")


def opencv_sign_page():
    _render_cv_module_page("sign")


def opencv_palm_page():
    _render_cv_module_page("palm")
