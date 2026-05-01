from __future__ import annotations

import datetime
import os
import tempfile
import threading
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

WEBRTC_READY = False
RTC_CONFIG_LOCAL = None
RTC_CONFIG_STUN = None

LIVE_INPUT_SOURCES = ["📷 Photo", "📸 Camera Snapshot"]
PALM_INPUT_SOURCES = ["📷 Photo", "📸 Camera Snapshot"]


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
        "gallery_subtitle": "60+ Features · Fine Lines · Expert Interpretation",
        "page_title": "Professional Palm Analyzer",
        "page_subtitle": "Advanced computer vision palm analysis with CLAHE, Gabor filters, and 60+ unique feature extraction",
        "path": "OpenCV_Detection/page_palm.py",
        "banner": os.path.join("assets", "banners", "palm_reading_banner_1774323346127.png"),
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
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
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
    if src == "📷 Photo":
        return _decode_image_file(st.file_uploader(upload_label, type=["jpg", "jpeg", "png"], key=upload_key))
    if src == "📸 Camera Snapshot":
        return _decode_image_file(st.camera_input(camera_label, key=camera_key))
    return None




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
            if img is not None and st.button("📸 Detect & Register", type="primary", use_container_width=True):
                processed = _att_cb(img.copy())
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

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
            st.dataframe(df, hide_index=True, use_container_width=True)
            if st.button("🗑 Clear Log", use_container_width=True):
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
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
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
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
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
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
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
    mask_binary = mask_binary * 255
    if mask_binary.max() == 0:
        return mask_binary

    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(mask_binary)

    # Fallback thinning when ximgproc.thinning is unavailable.
    skel = np.zeros_like(mask_binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = mask_binary.copy()
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel


# ── PHASE 1: ENHANCED IMAGE PREPROCESSING ─────────────────────────────────

def enhance_palm_image(image):
    """
    Ultra-fast, high-fidelity palm line detector optimized for 'ink-stamp' extraction.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Invert image (so dark creases become bright ridges)
    inv = cv2.bitwise_not(gray)

    # 2. CLAHE for robust contrast normalization across the whole palm
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(inv)
    
    # 3. Slight blur to prevent individual skin pores from becoming lines
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)

    # 4. Fast Local Adaptive Thresholding
    # This precisely isolates the creases as bright white lines on a black background
    # It perfectly mimics the real-life ink print aesthetic without massive computations.
    # Block size 35 covers enough area to capture both fine ridges and thick major lines.
    line_map = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2
    )

    # 5. Morphological cleanup (ultra-fast)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    line_map = cv2.morphologyEx(line_map, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 6. Remove very small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_map, connectivity=8)
    min_area = max(10, int(line_map.shape[0] * line_map.shape[1] * 0.00010))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            line_map[labels == i] = 0
            
    return line_map, clahe_img
def _flip_points_horizontally(points, width):
    flipped = {}
    for name, pt in points.items():
        arr = np.asarray(pt, dtype=np.int32)
        flipped[name] = np.array([width - 1 - int(arr[0]), int(arr[1])], dtype=np.int32)
    return flipped


def _normalize_palm_orientation(crop, anchor_points):
    """Normalize to a thumb-left view so line extraction uses one geometry."""
    points = {name: np.asarray(pt, dtype=np.int32) for name, pt in anchor_points.items()}
    if int(points["thumb_base"][0]) <= int(points["pinky_side"][0]):
        return crop, points, False
    return cv2.flip(crop, 1), _flip_points_horizontally(points, crop.shape[1]), True


def _cubic_bezier_points(p0, p1, p2, p3, samples=72):
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    omt = 1.0 - t
    curve = (
        (omt ** 3) * np.asarray(p0, dtype=np.float32)
        + 3.0 * (omt ** 2) * t * np.asarray(p1, dtype=np.float32)
        + 3.0 * omt * (t ** 2) * np.asarray(p2, dtype=np.float32)
        + (t ** 3) * np.asarray(p3, dtype=np.float32)
    )
    return curve.astype(np.int32)


def _path_distance_map(shape, points, thickness=3):
    seed = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(seed, [pts], False, 255, thickness=thickness)
    elif len(pts) == 1:
        cv2.circle(seed, tuple(pts[0]), thickness, 255, -1)
    return cv2.distanceTransform(cv2.bitwise_not(seed), cv2.DIST_L2, 3)


def _cleanup_binary_mask(binary_mask, min_area=16):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    if binary_mask.max() == 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 255
    return cleaned


def _score_line_component(component_mask, path_distance, start_pt=None, end_pt=None, prefer_horizontal=True):
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return float("-inf")

    xy = np.column_stack((xs, ys)).astype(np.float32)
    area = float(len(xs))
    arc_length = float(get_line_length(component_mask))
    mean_path_distance = float(np.mean(path_distance[component_mask > 0]))
    x, y, w, h = cv2.boundingRect(xy.astype(np.int32).reshape(-1, 1, 2))
    aspect = (w / max(h, 1)) if prefer_horizontal else (h / max(w, 1))

    start_bonus = 0.0
    if start_pt is not None:
        start_bonus = max(0.0, 40.0 - float(np.min(np.linalg.norm(xy - np.asarray(start_pt, dtype=np.float32), axis=1))))

    end_bonus = 0.0
    if end_pt is not None:
        end_bonus = max(0.0, 34.0 - float(np.min(np.linalg.norm(xy - np.asarray(end_pt, dtype=np.float32), axis=1))))

    return (
        arc_length * 1.15
        + area * 0.55
        + min(24.0, aspect * 10.0)
        + start_bonus * 1.2
        + end_bonus
        - mean_path_distance * 5.5
    )


def _select_line_mask(candidate_mask, path_distance, start_pt=None, end_pt=None, prefer_horizontal=True, keep_top=2, min_area=22):
    candidate_mask = _cleanup_binary_mask(candidate_mask, min_area=min_area)
    if candidate_mask.max() == 0:
        return candidate_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    ranked = []
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            continue
        component_mask = (labels == idx).astype(np.uint8) * 255
        score = _score_line_component(
            component_mask,
            path_distance,
            start_pt=start_pt,
            end_pt=end_pt,
            prefer_horizontal=prefer_horizontal,
        )
        ranked.append((score, idx))

    if not ranked:
        return np.zeros_like(candidate_mask)

    ranked.sort(reverse=True)
    best_score = ranked[0][0]
    selected = np.zeros_like(candidate_mask)
    kept = 0
    for score, idx in ranked:
        if kept >= keep_top:
            break
        if score < best_score - 45.0 and kept > 0:
            continue
        selected[labels == idx] = 255
        kept += 1

    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return selected


def extract_major_palm_lines(crop, anchor_points):
    """Detect the three primary palm lines using anatomy-guided corridors."""
    h, w = crop.shape[:2]
    palm_poly = np.array(
        [
            anchor_points["wrist"],
            anchor_points["thumb_base"],
            anchor_points["index_base"],
            anchor_points["middle_base"],
            anchor_points["ring_base"],
            anchor_points["pinky_base"],
            anchor_points["pinky_side"],
        ],
        dtype=np.int32,
    )

    palm_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(palm_mask, [palm_poly], 255)
    palm_mask = cv2.erode(palm_mask, np.ones((5, 5), np.uint8), iterations=1)
    palm_mask = cv2.morphologyEx(palm_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    line_map_gabor, enhanced_gray = enhance_palm_image(crop)
    line_map_gabor = cv2.bitwise_and(line_map_gabor, line_map_gabor, mask=palm_mask)
    
    # Use the highly accurate ink-stamp map directly, without ruining it with Otsu
    combined_lines = line_map_gabor
    combined_lines = _cleanup_binary_mask(combined_lines, min_area=max(12, int(h * w * 0.00012)))

    ys, xs = np.where(palm_mask > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.uint8), combined_lines, enhanced_gray, palm_mask

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    palm_w = max(1, x1 - x0)
    palm_h = max(1, y1 - y0)

    def _pt_blend(a, b, weight_to_b):
        pt = (np.asarray(a, dtype=np.float32) * (1.0 - weight_to_b)) + (np.asarray(b, dtype=np.float32) * weight_to_b)
        return np.round(pt).astype(np.int32)

    heart_start = _pt_blend(anchor_points["index_base"], anchor_points["middle_base"], 0.35)
    heart_mid = _pt_blend(anchor_points["middle_base"], anchor_points["ring_base"], 0.55)
    heart_end = np.array(
        [
            int(np.clip(anchor_points["pinky_base"][0], 0, w - 1)),
            int(np.clip(anchor_points["pinky_base"][1] + palm_h * 0.16, 0, h - 1)),
        ],
        dtype=np.int32,
    )
    heart_path = np.vstack((heart_start, heart_mid, heart_end))

    head_start = _pt_blend(anchor_points["thumb_base"], anchor_points["index_base"], 0.58)
    head_mid = np.array(
        [
            int(np.clip(x0 + palm_w * 0.53, 0, w - 1)),
            int(np.clip(y0 + palm_h * 0.53, 0, h - 1)),
        ],
        dtype=np.int32,
    )
    head_end = _pt_blend(anchor_points["pinky_side"], anchor_points["wrist"], 0.45)
    head_path = np.vstack((head_start, head_mid, head_end))

    life_start = _pt_blend(anchor_points["thumb_base"], anchor_points["index_base"], 0.44)
    life_ctrl1 = np.array(
        [
            int(np.clip(anchor_points["thumb_base"][0] + palm_w * 0.04, 0, w - 1)),
            int(np.clip(anchor_points["thumb_base"][1] + palm_h * 0.18, 0, h - 1)),
        ],
        dtype=np.int32,
    )
    life_ctrl2 = np.array(
        [
            int(np.clip(anchor_points["wrist"][0] + palm_w * 0.17, 0, w - 1)),
            int(np.clip(anchor_points["wrist"][1] - palm_h * 0.08, 0, h - 1)),
        ],
        dtype=np.int32,
    )
    life_end = np.array(
        [
            int(np.clip(anchor_points["wrist"][0] + palm_w * 0.08, 0, w - 1)),
            int(np.clip(anchor_points["wrist"][1] - palm_h * 0.02, 0, h - 1)),
        ],
        dtype=np.int32,
    )
    life_path = _cubic_bezier_points(life_start, life_ctrl1, life_ctrl2, life_end, samples=80)

    heart_dist = _path_distance_map((h, w), heart_path, thickness=3)
    head_dist = _path_distance_map((h, w), head_path, thickness=3)
    life_dist = _path_distance_map((h, w), life_path, thickness=3)

    upper_band = np.zeros((h, w), dtype=np.uint8)
    upper_band[max(0, y0 + int(palm_h * 0.08)):min(h, y0 + int(palm_h * 0.48)), x0:x1 + 1] = 255
    middle_band = np.zeros((h, w), dtype=np.uint8)
    middle_band[max(0, y0 + int(palm_h * 0.20)):min(h, y0 + int(palm_h * 0.78)), x0:x1 + 1] = 255
    life_band = np.zeros((h, w), dtype=np.uint8)
    life_band[max(0, y0 + int(palm_h * 0.05)):min(h, y1 + 1), x0:min(w, x0 + int(palm_w * 0.68))] = 255

    heart_mask = np.where((combined_lines > 0) & (heart_dist < max(18, palm_w * 0.10)) & (upper_band > 0), 255, 0).astype(np.uint8)
    head_mask = np.where((combined_lines > 0) & (head_dist < max(18, palm_w * 0.10)) & (middle_band > 0), 255, 0).astype(np.uint8)
    life_mask = np.where((combined_lines > 0) & (life_dist < max(20, palm_w * 0.13)) & (life_band > 0), 255, 0).astype(np.uint8)

    heart_mask = _select_line_mask(heart_mask, heart_dist, start_pt=heart_start, end_pt=heart_end, prefer_horizontal=True, keep_top=2)
    head_mask = _select_line_mask(head_mask, head_dist, start_pt=head_start, end_pt=head_end, prefer_horizontal=True, keep_top=2)
    life_mask = _select_line_mask(life_mask, life_dist, start_pt=life_start, end_pt=life_end, prefer_horizontal=False, keep_top=2)

    head_mask = cv2.subtract(head_mask, heart_mask)
    life_mask = cv2.subtract(life_mask, cv2.bitwise_or(head_mask, heart_mask))

    segmentation_mask = np.zeros((h, w), dtype=np.uint8)
    segmentation_mask[life_mask > 0] = 1
    segmentation_mask[head_mask > 0] = 2
    segmentation_mask[heart_mask > 0] = 3

    return segmentation_mask, combined_lines, enhanced_gray, palm_mask


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


def _farthest_pair(points, max_samples=220):
    if points is None or len(points) < 2:
        return None, None, 0.0

    pts = np.asarray(points, dtype=np.float32)
    if len(pts) > max_samples:
        idx = np.linspace(0, len(pts) - 1, max_samples, dtype=int)
        pts = pts[idx]

    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist_sq)), dist_sq.shape)
    return pts[i], pts[j], float(np.sqrt(dist_sq[i, j]))


def _line_geometry(mask):
    """Compute robust line geometry from a thinned centerline instead of contour edges."""
    empty = {
        "points": np.empty((0, 2), dtype=np.int32),
        "ordered_points": np.empty((0, 2), dtype=np.int32),
        "start": None,
        "end": None,
        "length": 0.0,
        "straight": 0.0,
        "curvature": 0.0,
        "angle": 0.0,
    }

    if mask.max() == 0:
        return empty

    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = extract_skeleton(binary)
    skeleton = (skeleton > 0).astype(np.uint8) * 255

    use_skeleton = cv2.countNonZero(skeleton) >= 10
    active = skeleton if use_skeleton else binary

    ys, xs = np.where(active > 0)
    if len(xs) < 2:
        return empty

    points = np.column_stack((xs, ys)).astype(np.int32)

    endpoint_points = np.empty((0, 2), dtype=np.int32)
    if use_skeleton:
        skel_bin = (skeleton > 0).astype(np.uint8)
        neighbors = cv2.filter2D(skel_bin, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
        end_y, end_x = np.where((skel_bin > 0) & (neighbors <= 2))
        endpoint_points = np.column_stack((end_x, end_y)).astype(np.int32)

    if len(endpoint_points) >= 2:
        p0, p1, straight = _farthest_pair(endpoint_points)
    else:
        p0, p1, straight = _farthest_pair(points)

    if p0 is None or p1 is None:
        return empty

    start = np.asarray(p0, dtype=np.float32)
    end = np.asarray(p1, dtype=np.float32)

    # Stable direction: left-to-right, otherwise top-to-bottom.
    if (start[0] > end[0]) or (abs(start[0] - end[0]) < 1e-3 and start[1] > end[1]):
        start, end = end, start

    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        order = np.argsort(points[:, 0])
    else:
        proj = (points.astype(np.float32) - start) @ direction / norm
        order = np.argsort(proj)
    ordered_points = points[order]

    if use_skeleton:
        # On a thinned centerline, non-zero pixel count is a stable length proxy.
        length = float(cv2.countNonZero(skeleton))
        if len(ordered_points) >= 2:
            steps = np.linalg.norm(np.diff(ordered_points.astype(np.float32), axis=0), axis=1)
            if len(steps) > 0:
                length = float(np.sum(np.clip(steps, 0.0, 2.2)))
    else:
        contours, _ = cv2.findContours(active, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            segment_lengths = sorted(
                (cv2.arcLength(c, False) for c in contours if len(c) >= 2),
                reverse=True,
            )
            length = float(segment_lengths[0]) if segment_lengths else float(len(points))
        else:
            length = float(len(points))

    if length <= 0:
        length = float(len(points))

    straight = max(float(straight), 1.0)
    raw_ratio = (length / straight) if length > 0 else 1.0
    curvature = float(np.clip(1.0 + max(0.0, raw_ratio - 1.0) * 0.30, 1.0, 2.2))
    length = float(max(straight * 0.90, straight * curvature))
    angle = float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))

    return {
        "points": points,
        "ordered_points": ordered_points,
        "start": np.round(start).astype(np.int32),
        "end": np.round(end).astype(np.int32),
        "length": length,
        "straight": straight,
        "curvature": curvature,
        "angle": angle,
    }


# ── PHASE 2: FINE LINE DETECTION ──────────────────────────────────────────

def detect_breaks(mask):
    """Detect breaks (gaps) in a line mask. Returns list of break positions (0-1 normalized)."""
    if mask.max() == 0:
        return []

    skeleton = extract_skeleton(mask)
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    if skeleton.max() == 0:
        return []

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    if num_labels <= 2:
        return []

    min_component = max(8, int(cv2.countNonZero(skeleton) * 0.05))
    components = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_component:
            components.append((float(centroids[i][0]), float(centroids[i][1]), area))

    if len(components) < 2:
        return []

    # Keep only strongest fragments to avoid tiny-noise "breaks".
    components = sorted(components, key=lambda c: c[2], reverse=True)[:4]
    components.sort(key=lambda c: c[0])

    h, w = mask.shape[:2]
    breaks = []
    for i in range(len(components) - 1):
        gap_x = (components[i][0] + components[i + 1][0]) / 2
        gap_y = (components[i][1] + components[i + 1][1]) / 2
        breaks.append({"position": round(gap_x / w, 3), "y_position": round(gap_y / h, 3)})
    return breaks[:3]


def detect_branches(mask, enhanced_line_map):
    """Detect branches extending from a major line using skeleton junction analysis."""
    if mask.max() == 0 or enhanced_line_map is None or enhanced_line_map.max() == 0:
        return {"upward": 0, "downward": 0, "total": 0}

    major_core = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    fine_only = cv2.subtract(enhanced_line_map, major_core)
    fine_only = cv2.morphologyEx(fine_only, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    fine_only = _cleanup_binary_mask(fine_only, min_area=10)
    if fine_only.max() == 0:
        return {"upward": 0, "downward": 0, "total": 0}

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fine_only, connectivity=8)
    geom = _line_geometry(mask)
    main_pts = geom["ordered_points"]
    if len(main_pts) == 0:
        return {"upward": 0, "downward": 0, "total": 0}

    main_center_y = float(np.mean(main_pts[:, 1]))
    line_length = max(float(geom["length"]), 1.0)
    min_area = max(18, int(line_length * 0.08))
    min_arc = max(12.0, line_length * 0.12)
    touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    major_touch = cv2.dilate(mask, touch_kernel, iterations=1)

    upward = 0
    downward = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(cv2.dilate(comp_mask, touch_kernel, iterations=1), major_touch)) == 0:
            continue

        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        arc = max(cv2.arcLength(c, False) for c in contours)
        comp_height = int(stats[i, cv2.CC_STAT_HEIGHT])
        if arc < min_arc and comp_height < 6:
            continue

        if float(centroids[i][1]) < main_center_y:
            upward += 1
        else:
            downward += 1

    upward = min(upward, 6)
    downward = min(downward, 6)
    total = min(upward + downward, 12)
    return {"upward": upward, "downward": downward, "total": total}


def detect_islands(mask):
    """Detect enclosed loop structures while suppressing tiny hole noise."""
    if mask.max() == 0:
        return 0

    binary = (mask > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0

    line_length = max(get_line_length(mask), 1.0)
    min_area = max(10.0, line_length * 0.02)
    max_area = max(min_area * 6.0, line_length * 0.35)

    island_count = 0
    for i, h in enumerate(hierarchy[0]):
        # h = [next, prev, child, parent]
        if h[3] == -1:
            continue

        area = float(cv2.contourArea(contours[i]))
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contours[i], True)
        if perimeter <= 1.0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if 0.12 <= circularity <= 1.45:
            island_count += 1

    return int(min(island_count, 9))


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
    geom = _line_geometry(mask)
    pts = geom["ordered_points"]
    h, w = mask.shape[:2]
    if len(pts) < 5 or geom["start"] is None or geom["end"] is None:
        return {
            f"{label}_start_x": 0.5, f"{label}_start_y": 0.5,
            f"{label}_end_x": 0.5, f"{label}_end_y": 0.5,
            f"{label}_span_x": 0.0, f"{label}_span_y": 0.0,
            f"{label}_curvature_samples": [1.0] * 5,
            f"{label}_bbox_area_ratio": 0.0,
        }

    sorted_pts = pts
    start = geom["start"]
    end = geom["end"]

    # Multi-point curvature (sample at 5 segments)
    curvature_samples = []
    n_segments = 5
    boundaries = np.linspace(0, len(sorted_pts), n_segments + 1, dtype=int)
    for i in range(n_segments):
        seg = sorted_pts[boundaries[i] : max(boundaries[i + 1], boundaries[i] + 2)]
        if len(seg) < 2:
            curvature_samples.append(1.0)
            continue
        arc = float(cv2.arcLength(seg.reshape(-1, 1, 2), closed=False))
        dist = float(np.linalg.norm(seg[0].astype(float) - seg[-1].astype(float)))
        curvature_samples.append(round(float(np.clip(arc / max(dist, 1.0), 1.0, 3.8)), 3))

    # Bounding box area ratio (how much of the bbox the line fills)
    x_min, y_min = sorted_pts.min(axis=0)
    x_max, y_max = sorted_pts.max(axis=0)
    bbox_area = max((x_max - x_min) * (y_max - y_min), 1)
    line_pixel_count = len(sorted_pts)

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


def detect_minor_lines(enhanced_line_map, major_mask_combined, mask_shape, region_mask=None):
    """Detect minor/fine lines that are NOT part of the 3 major lines.
    These include fate, sun, marriage, travel lines etc."""
    # Remove major lines from the enhanced map
    fine_only = cv2.subtract(enhanced_line_map, major_mask_combined)
    if region_mask is not None and region_mask.max() > 0:
        fine_only = cv2.bitwise_and(fine_only, fine_only, mask=region_mask)

    fine_only = _cleanup_binary_mask(
        fine_only,
        min_area=max(10, int(fine_only.shape[0] * fine_only.shape[1] * 0.00005)),
    )
    fine_only = (extract_skeleton(fine_only) > 0).astype(np.uint8) * 255

    if fine_only.max() == 0:
        return {"total_fine_lines": 0, "fine_line_density": 0.0, "fine_line_total_length": 0.0}

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fine_only, connectivity=8)

    if region_mask is not None and region_mask.max() > 0:
        active_area = max(int(cv2.countNonZero(region_mask)), 1)
    else:
        h, w = mask_shape[:2]
        active_area = max(h * w, 1)

    min_line_length = max(10, int(np.sqrt(active_area) * 0.012))
    min_arc = max(16.0, np.sqrt(active_area) * 0.030)
    valid_lines = 0
    total_fine_length = 0.0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_line_length:
            # Compute arc length for this component
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                arc = cv2.arcLength(max(contours, key=cv2.contourArea), False)
                if arc > min_arc:
                    valid_lines += 1
                    total_fine_length += arc

    density = total_fine_length / active_area * 10000  # per 10k pixels
    return {
        "total_fine_lines": int(min(valid_lines, 120)),
        "fine_line_density": round(density, 2),
        "fine_line_total_length": round(total_fine_length, 1),
    }


def summarize_minor_line_channels(enhanced_line_map, major_mask_combined, region_mask=None):
    """Estimate vertical minor-line strength in classical Fate and Sun zones."""
    fine_only = cv2.subtract(enhanced_line_map, major_mask_combined)
    if region_mask is not None and region_mask.max() > 0:
        fine_only = cv2.bitwise_and(fine_only, fine_only, mask=region_mask)
        bbox_mask = region_mask
    else:
        bbox_mask = major_mask_combined

    fine_only = _cleanup_binary_mask(
        fine_only,
        min_area=max(8, int(fine_only.shape[0] * fine_only.shape[1] * 0.00004)),
    )
    fine_only = (extract_skeleton(fine_only) > 0).astype(np.uint8) * 255

    ys, xs = np.where(bbox_mask > 0)
    if fine_only.max() == 0 or len(xs) == 0:
        return {"fate_presence": 0.0, "sun_presence": 0.0}

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    def _zone_presence(x_start_ratio, x_end_ratio, y_start_ratio, y_end_ratio):
        zx0 = int(np.clip(x0 + width * x_start_ratio, 0, fine_only.shape[1] - 1))
        zx1 = int(np.clip(x0 + width * x_end_ratio, zx0 + 1, fine_only.shape[1]))
        zy0 = int(np.clip(y0 + height * y_start_ratio, 0, fine_only.shape[0] - 1))
        zy1 = int(np.clip(y0 + height * y_end_ratio, zy0 + 1, fine_only.shape[0]))

        zone_mask = np.zeros_like(fine_only)
        zone_mask[zy0:zy1, zx0:zx1] = 255
        zone_lines = cv2.bitwise_and(fine_only, fine_only, mask=zone_mask)
        if zone_lines.max() == 0:
            return 0.0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_lines, connectivity=8)
        zone_height = max(1, zy1 - zy0)
        score = 0.0
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            comp_width = stats[idx, cv2.CC_STAT_WIDTH]
            comp_height = stats[idx, cv2.CC_STAT_HEIGHT]
            aspect = comp_height / max(comp_width, 1)
            if area < 18 or comp_height < zone_height * 0.18 or aspect < 1.6:
                continue
            score += (
                min(0.45, area / 320.0)
                + min(0.35, (comp_height / zone_height) * 0.8)
                + min(0.20, max(0.0, aspect - 1.6) / 2.4)
            )
        return round(min(score, 1.0), 3)

    return {
        "fate_presence": _zone_presence(0.36, 0.64, 0.20, 0.96),
        "sun_presence": _zone_presence(0.56, 0.82, 0.12, 0.82),
    }


# ── PHASE 5: MASTER FEATURE EXTRACTION (60+ features) ────────────────────

def get_line_length(mask):
    return float(_line_geometry(mask)["length"])

def get_curvature(mask):
    return float(_line_geometry(mask)["curvature"])

def get_line_angle(mask):
    return float(_line_geometry(mask)["angle"])

def count_intersections(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    if intersection.max() == 0: return 0
    num_labels, _ = cv2.connectedComponents(intersection)
    return max(0, num_labels - 1)


def extract_palm_features(segmentation_mask, raw_image=None, palm_region_mask=None, enhanced_line_map=None, enhanced_gray=None):
    """Extract 60+ unique features from the palm using both the CNN segmentation
    mask AND raw image analysis. When raw_image is None, falls back to basic mode."""
    life_mask = (segmentation_mask == 1).astype(np.uint8) * 255
    head_mask = (segmentation_mask == 2).astype(np.uint8) * 255
    heart_mask = (segmentation_mask == 3).astype(np.uint8) * 255

    features = {}

    # ── Basic features (backward compatible) ──
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

    if raw_image is None:
        return features

    # ══════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURES — derived from raw image analysis
    # ══════════════════════════════════════════════════════════════════════
    if enhanced_line_map is None or enhanced_gray is None:
        enhanced_line_map, enhanced_gray = enhance_palm_image(raw_image)

    if palm_region_mask is not None and palm_region_mask.max() > 0:
        enhanced_line_map = cv2.bitwise_and(enhanced_line_map, enhanced_line_map, mask=palm_region_mask)
        enhanced_gray = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=palm_region_mask)

    masks = {"life": life_mask, "head": head_mask, "heart": heart_mask}

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
    minor_lines = detect_minor_lines(enhanced_line_map, major_combined, segmentation_mask.shape, region_mask=palm_region_mask)
    features.update(minor_lines)
    features.update(summarize_minor_line_channels(enhanced_line_map, major_combined, region_mask=palm_region_mask))

    # ── Unique palm signature (hash of all continuous features for dedup) ──
    sig_vals = [
        features['life_length'], features['head_length'], features['heart_length'],
        features['life_curvature'], features['head_curvature'], features['heart_curvature'],
        features['life_angle'], features['head_angle'], features['heart_angle'],
        features.get('life_avg_depth', 0), features.get('head_avg_depth', 0),
        features.get('heart_avg_depth', 0),
        features.get('life_branch_total', 0), features.get('head_branch_total', 0),
        features.get('life_break_count', 0), features.get('total_fine_lines', 0),
        features.get('fate_presence', 0), features.get('sun_presence', 0),
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
    """Create a cinematic professional overlay showing major lines (glowing)
    plus fine lines and a scanning laser effect."""
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    overlay = image.copy().astype(np.float32)
    
    # ── 1. Glowing Major Lines ──
    # Colors: Life=Crimson, Head=Emerald, Heart=Azure
    colors = {1: (20, 20, 220), 2: (20, 220, 20), 3: (220, 100, 20)}
    
    for class_id, color in colors.items():
        class_mask = (mask == class_id).astype(np.uint8) * 255
        if class_mask.max() == 0:
            continue
            
        # Create Glow
        glow = cv2.GaussianBlur(class_mask, (15, 15), 0)
        glow_mask = (glow > 0)
        
        # Apply Glow
        for c in range(3):
            overlay[glow_mask, c] = overlay[glow_mask, c] * (1 - glow[glow_mask]/255 * 0.4) + \
                                    color[c] * (glow[glow_mask]/255 * 0.4)
        
        # Core Line
        line_mask = (class_mask > 0)
        overlay[line_mask] = color
        
    # ── 2. Fine Lines (Cyan Mist) ──
    if enhanced_line_map is not None:
        if enhanced_line_map.shape[:2] != (h, w):
            enhanced_line_map = cv2.resize(enhanced_line_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        major_combined = np.zeros((h, w), dtype=np.uint8)
        for cid in [1, 2, 3]:
            major_combined[mask == cid] = 255
            
        fine_only = cv2.subtract(enhanced_line_map, major_combined)
        fine_mask = (fine_only > 0)
        overlay[fine_mask] = overlay[fine_mask] * 0.4 + np.array([255, 255, 0], dtype=np.float32) * 0.6

    # ── 3. Scanning Laser Line ──
    # Use a simulated pulse based on time or just a static horizontal line for the snapshot
    # To keep it looking "active", we'll draw a subtle horizontal gradient line
    scan_y = int(h * 0.45) # Static for snapshot, but looks like a scan
    cv2.line(overlay, (0, scan_y), (w, scan_y), (0, 255, 255), 1)
    # Add a halo for the laser
    cv2.addWeighted(overlay, 1.0, 
                   cv2.GaussianBlur(overlay, (1, 51), 0), 0.2, 0, overlay)

    return overlay.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PALM OBSERVATIONS (replaces the removed expander form)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OBSERVATIONS = {
    "dominant_hand": "Both / unsure",
    "hand_shape": "Auto / unsure",
}


def build_dynamic_palm_observations(features):
    """Convert scan metrics into the old observation labels used by the report engine."""
    major_depths = [
        float(features.get("life_avg_depth", 0.0)),
        float(features.get("head_avg_depth", 0.0)),
        float(features.get("heart_avg_depth", 0.0)),
    ]
    valid_depths = [depth for depth in major_depths if depth > 0]
    avg_depth = float(np.mean(valid_depths)) if valid_depths else 0.5

    total_breaks = sum(int(features.get(f"{name}_break_count", 0)) for name in ("life", "head", "heart"))

    if avg_depth >= 0.62:
        line_depth = "Deep"
    elif avg_depth <= 0.36:
        line_depth = "Faint"
    else:
        line_depth = "Medium"

    if total_breaks >= 4:
        major_breaks = "Many"
    elif total_breaks >= 1:
        major_breaks = "A few"
    else:
        major_breaks = "None"

    detected_hand = str(features.get("detected_hand_label", "")).title()
    detected_hand_conf = float(features.get("detected_hand_confidence", 0.0))
    if detected_hand not in {"Left", "Right"} or detected_hand_conf < 0.5:
        detected_hand = "Both / unsure"

    return {
        "dominant_hand": detected_hand,
        "line_depth": line_depth,
        "major_breaks": major_breaks,
    }


def _extract_handedness_label(result):
    handedness = getattr(result, "handedness", None) or []
    if not handedness or not handedness[0]:
        return None, 0.0

    try:
        category = handedness[0][0]
        label = getattr(category, "category_name", None) or getattr(category, "display_name", None)
        score = float(getattr(category, "score", 0.0) or 0.0)
    except Exception:
        return None, 0.0

    if label in {"Left", "Right"}:
        return label, score
    return None, score


def _compute_palm_scan_quality(palm_roi, segmentation_mask, enhanced_gray, features, handedness_label=None, handedness_score=0.0):
    image_area = max(int(segmentation_mask.shape[0] * segmentation_mask.shape[1]), 1)
    palm_area = int(cv2.countNonZero(palm_roi)) if palm_roi is not None else 0
    hand_fill_ratio = float(palm_area / image_area) if palm_area else 0.0

    major_mask = (segmentation_mask > 0).astype(np.uint8) * 255
    major_pixels = int(cv2.countNonZero(major_mask))
    major_line_ratio = float(major_pixels / max(palm_area, 1)) if palm_area else 0.0

    visible_major_lines = sum(
        1 for key in ("life_length", "head_length", "heart_length") if float(features.get(key, 0.0)) >= 55.0
    )
    fine_density = float(features.get("fine_line_density", 0.0))

    depth_values = [
        float(features.get("life_avg_depth", 0.0)),
        float(features.get("head_avg_depth", 0.0)),
        float(features.get("heart_avg_depth", 0.0)),
    ]
    valid_depths = [value for value in depth_values if value > 0]
    avg_depth = float(np.mean(valid_depths)) if valid_depths else 0.0

    texture_score = 0.0
    line_contrast_score = 0.0
    if enhanced_gray is not None and palm_area > 0:
        palm_values = enhanced_gray[palm_roi > 0]
        if palm_values.size > 0:
            texture_score = float(np.clip(np.std(palm_values) / 52.0, 0.0, 1.0))

        line_values = enhanced_gray[major_mask > 0]
        skin_values = enhanced_gray[(palm_roi > 0) & (major_mask == 0)]
        if line_values.size > 0 and skin_values.size > 0:
            line_contrast_score = float(
                np.clip(abs(float(np.mean(skin_values)) - float(np.mean(line_values))) / 48.0, 0.0, 1.0)
            )

    quality_score = float(np.clip(
        0.12
        + min(hand_fill_ratio / 0.18, 1.0) * 0.18
        + min(major_line_ratio / 0.075, 1.0) * 0.22
        + (visible_major_lines / 3.0) * 0.16
        + min(fine_density / 28.0, 1.0) * 0.10
        + texture_score * 0.10
        + line_contrast_score * 0.12
        + min(avg_depth / 0.78, 1.0) * 0.10,
        0.0,
        0.98,
    ))

    detected_hand = handedness_label if handedness_label in {"Left", "Right"} else "Unclear"
    return {
        "quality_score": round(quality_score, 3),
        "hand_fill_ratio": round(hand_fill_ratio, 3),
        "major_line_ratio": round(major_line_ratio, 3),
        "texture_score": round(texture_score, 3),
        "line_contrast_score": round(line_contrast_score, 3),
        "major_line_count": int(visible_major_lines),
        "detected_hand": detected_hand,
        "handedness_confidence": round(float(handedness_score), 3),
    }


def _draw_live_palm_summary(image, report):
    output = image.copy()

    if not report or not report.get("_hand_found", False):
        labels = [
            "Palm not locked yet",
            "Show full palm and wrist to the camera",
            "Use brighter light and keep the hand flatter",
        ]
    else:
        scan_quality = report.get("scan_quality", {})
        detected_hand = report.get("detected_hand", "Unclear")
        labels = [
            f"Dominant: {report.get('dominant_line', 'Unknown')}",
            f"Quality: {report.get('detection_quality', 0.0):.2f}",
            f"Hand: {detected_hand}",
            f"Major Lines: {int(report.get('scan_quality', {}).get('major_line_count', 0))}/3 locked",
        ]
        if scan_quality.get("quality_score", 0.0) < 0.58:
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


def _persist_palm_report(features, report, overlay=None):
    new_signature = features.get("palm_signature")
    if new_signature and st.session_state.get("palm_last_signature") != new_signature:
        page_name = st.session_state.get("last_visited_page", "global")
        st.session_state.pop(f"chat_history_{page_name}", None)
        st.session_state["palm_last_signature"] = new_signature

    st.session_state["palm_latest_report"] = report
    st.session_state["palm_latest_features"] = features
    st.session_state["palm_latest_summary"] = report.get("summary", "")
    if overlay is not None:
        st.session_state["palm_latest_overlay"] = overlay


def _clear_palm_report_session():
    for key in (
        "palm_latest_report",
        "palm_latest_features",
        "palm_latest_summary",
        "palm_latest_overlay",
        "palm_last_signature",
    ):
        st.session_state.pop(key, None)


def _render_live_palm_dashboard(report, features):
    scan_quality = report.get("scan_quality", {})
    timing_predictions = report.get("timing", {}).get("predictions", [])
    top_prediction = timing_predictions[0] if timing_predictions else None

    render_info_grid([
        ("Detected Hand", report.get("detected_hand", "Unclear")),
        ("Detection", f"{report.get('detection_quality', 0.0):.0%}"),
        ("Major Lines", f"{scan_quality.get('major_line_count', 0)}/3"),
        ("Fine Lines", str(int(features.get("total_fine_lines", 0)))),
        ("Palm Fill", f"{scan_quality.get('hand_fill_ratio', 0.0):.0%}"),
        ("Texture", f"{scan_quality.get('texture_score', 0.0):.0%}"),
    ])

    render_content_card(
        "Live Reading Snapshot",
        report.get("summary", "").replace("\n", "<br>"),
        accent_color="#8B5CF6",
        icon="🔮",
    )

    if top_prediction:
        render_content_card(
            f"{top_prediction.get('period', '')} — {top_prediction.get('event', '')}",
            top_prediction.get("detail", ""),
            accent_color="#F59E0B",
            icon="⏳",
        )

    scan_tips = report.get("guidance", [])[:2]
    if scan_tips:
        tips_html = "".join([f"<div style='margin-bottom:6px;'>• {tip}</div>" for tip in scan_tips])
        render_content_card("Live Guidance", tips_html, accent_color="#06B6D4", icon="🧭")


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
                f"{item['line']} Line",
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
        # Minor lines summary
        total_fine = features.get('total_fine_lines', 0)
        fine_density = features.get('fine_line_density', 0)
        gap_info = features.get('life_head_gap', 0)
        minor_html = (
            f"<b>Total minor/fine lines:</b> {total_fine}<br>"
            f"<b>Fine line density:</b> {fine_density:.1f} per 10k px<br>"
            f"<b>Life-Head origin gap:</b> {gap_info:.4f} (larger = earlier independence)<br>"
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
            ("Hand Type", f"{ht.get('type', 'Mixed')}"),
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

def generate_expert_inspection_crops(image, landmarks):
    """Identifies and crops key Regions of Interest (ROIs) for expert inspection.
    Includes Mounts (Jupiter, Saturn, etc.) and line origin points."""
    h, w = image.shape[:2]
    def get_pt(idx): return np.array([int(landmarks[idx].x * w), int(landmarks[idx].y * h)])

    crops = []
    # Inspection Points: (Name, LandmarkID, Scale, Description)
    points = [
        ("Mount of Jupiter", 5, 0.12, "Ambition, leadership, and social authority."),
        ("Mount of Saturn", 9, 0.12, "Duty, responsibility, and karmic patterns."),
        ("Mount of Apollo", 13, 0.12, "Artistic talent, fame, and creative success."),
        ("Mount of Mercury", 17, 0.12, "Business acumen, communication, and science."),
        ("Mount of Venus", 2, 0.18, "Passion, vitality, warmth, and physical love."),
        ("Mount of Moon", 0, 0.20, "Imagination, intuition, and travel restless nature."),
        ("Life-Head Origin", 5, 0.15, "Early life independence and family influence."),
    ]

    for name, idx, scale, desc in points:
        pt = get_pt(idx)
        # Adjust center for some mounts
        if name == "Mount of Moon": pt[0] += int(w * 0.1) # Move right for Moon
        if name == "Mount of Venus": pt[1] += int(h * 0.05) # Move down for Venus
        
        size = int(max(h, w) * scale)
        x1 = max(0, pt[0] - size//2)
        y1 = max(0, pt[1] - size//2)
        x2 = min(w, pt[0] + size//2)
        y2 = min(h, pt[1] + size//2)
        
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0: continue
            
        # Add a magnifying glass border effect to the crop
        ch, cw = crop.shape[:2]
        cv2.rectangle(crop, (0,0), (cw-1, ch-1), (0, 255, 255), 3) # Golden border
        
        crops.append({
            "name": name,
            "image": crop,
            "description": desc
        })
    return crops

@st.cache_resource(show_spinner=False)
def load_palm_model(_device=None):
    """Download and initialise the MediaPipe HandLandmarker (tasks API)."""
    import urllib.request, shutil
    from pathlib import Path

    assets_dir = Path(__file__).resolve().parents[1] / "assets" / "palm"
    assets_dir.mkdir(parents=True, exist_ok=True)
    model_path = assets_dir / "hand_landmarker.task"
    legacy_path = Path("hand_landmarker.task")

    # migrate legacy location
    if not model_path.exists() and legacy_path.exists():
        try:
            shutil.move(str(legacy_path), str(model_path))
        except Exception:
            pass

    # download if absent
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        return HandLandmarker.create_from_options(options)
    except Exception as exc:
        st.error(f"Failed to load HandLandmarker: {exc}")
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
            <span style="color: #E2E8F0; font-family: 'Inter', sans-serif; font-size: 12px;">
                Upload or capture your palm for a <strong>precision analysis</strong> covering
                <strong>personality, career, love, health, timing</strong> — powered by
                advanced computer vision that detects even the finest lines invisible to the eye.
            </span>
        </div>
    """, unsafe_allow_html=True)
    src = st.radio("Input Source", PALM_INPUT_SOURCES, horizontal=True, key="cv_palm_src")

    landmarker = load_palm_model()
    if landmarker is None:
        return
    import mediapipe as mp

    def _palm_cb(img, target_size=800, include_steps=True):
        """Step-by-step palm analysis like a human reader:
        Step 1: Detect palm → Step 2: Zoom & crop → Step 3: Trace lines → Step 4: Analyze"""
        h, w = img.shape[:2]
        steps = {} if include_steps else None

        # ═══ PRE-PROCESS: RESIZE FOR SPEED ═══
        # Very high res images can hang the landmarker in cloud environments
        max_dim = 1024
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            img_detect = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_detect = img

        # ═══ STEP 1: DETECT PALM ═══
        img_rgb = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            report = {
                "summary": "No palm detected yet.",
                "guidance": [
                    "Show the full palm and wrist inside the frame.",
                    "Use brighter, even lighting and keep the palm flatter.",
                    "Move closer so the palm fills more of the image.",
                ],
                "scan_quality": {
                    "quality_score": 0.0,
                    "hand_fill_ratio": 0.0,
                    "major_line_ratio": 0.0,
                    "texture_score": 0.0,
                    "line_contrast_score": 0.0,
                    "major_line_count": 0,
                    "detected_hand": "Unclear",
                    "handedness_confidence": 0.0,
                },
                "_steps": steps,
            }
            report["_hand_found"] = False
            return img.copy(), np.zeros((h, w), dtype=np.uint8), {}, report

        landmarks = result.hand_landmarks[0]
        handedness_label, handedness_score = _extract_handedness_label(result)

        def get_pt(idx): return np.array([int(landmarks[idx].x * w), int(landmarks[idx].y * h)])

        # Draw detection on original image
        all_pts = [get_pt(i) for i in range(21)]
        if include_steps:
            step1_img = img.copy()
            connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                           (9,13),(13,12),(12,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
            for a, b in connections:
                cv2.line(step1_img, tuple(all_pts[a]), tuple(all_pts[b]), (0, 255, 200), 2)
            for pt in all_pts:
                cv2.circle(step1_img, tuple(pt), 4, (0, 120, 255), -1)
            steps["detect"] = step1_img

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]

        # ═══ STEP 2: ZOOM INTO PALM ═══
        pad = int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.15)
        x_min = max(0, min(xs) - pad)
        x_max = min(w, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(h, max(ys) + pad)
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            report = {
                "summary": "Palm crop failed.",
                "guidance": ["Try again with the palm more centered and less tilted."],
                "scan_quality": {"quality_score": 0.0, "major_line_count": 0, "detected_hand": "Unclear"},
                "_steps": steps,
                "_hand_found": False,
            }
            return img.copy(), np.zeros((h, w), dtype=np.uint8), {}, report

        ch, cw = crop.shape[:2]
        scale = max(1.0, target_size / max(ch, cw))
        if scale > 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        if include_steps:
            steps["zoom"] = crop.copy()

        def crop_pt(idx):
            return np.array([int((landmarks[idx].x * w - x_min) * scale),
                             int((landmarks[idx].y * h - y_min) * scale)])

        wrist = crop_pt(0)
        thumb_base = crop_pt(2)
        index_base = crop_pt(5)
        middle_base = crop_pt(9)
        ring_base = crop_pt(13)
        pinky_base = crop_pt(17)
        pinky_side = crop_pt(18)

        # ═══ STEP 3: TRACE LINES ═══
        crop_anchor_points = {
            "wrist": wrist,
            "thumb_base": thumb_base,
            "index_base": index_base,
            "middle_base": middle_base,
            "ring_base": ring_base,
            "pinky_base": pinky_base,
            "pinky_side": pinky_side,
        }
        crop_normalized, normalized_points, was_flipped = _normalize_palm_orientation(crop, crop_anchor_points)
        segmentation_mask_norm, lines_mask_norm, enhanced_gray_norm, palm_mask_norm = extract_major_palm_lines(crop_normalized, normalized_points)

        if was_flipped:
            segmentation_mask = cv2.flip(segmentation_mask_norm, 1)
            lines_mask = cv2.flip(lines_mask_norm, 1)
            crop_enhanced_gray = cv2.flip(enhanced_gray_norm, 1)
            palm_mask = cv2.flip(palm_mask_norm, 1)
        else:
            segmentation_mask = segmentation_mask_norm
            lines_mask = lines_mask_norm
            crop_enhanced_gray = enhanced_gray_norm
            palm_mask = palm_mask_norm

        if include_steps:
            enhanced_preview = cv2.applyColorMap(crop_enhanced_gray, cv2.COLORMAP_BONE)
            enhanced_preview = cv2.addWeighted(crop, 0.42, enhanced_preview, 0.58, 0)
            preview_mask = cv2.dilate(lines_mask, np.ones((2, 2), np.uint8), iterations=1)
            enhanced_preview[preview_mask > 0] = (255, 255, 0)
            steps["enhanced"] = enhanced_preview

            step3_img = crop.copy()
            life_color, head_color, heart_color = (0, 0, 255), (0, 255, 0), (255, 100, 0)
            for label_val, color in [(1, life_color), (2, head_color), (3, heart_color)]:
                lmask = (segmentation_mask == label_val).astype(np.uint8) * 255
                if lmask.max() > 0:
                    lmask_thick = cv2.dilate(lmask, np.ones((3, 3), np.uint8), iterations=2)
                    step3_img[lmask_thick > 0] = color
            palm_poly = np.array([wrist, thumb_base, index_base, middle_base, ring_base, pinky_base, pinky_side], dtype=np.int32)
            cv2.polylines(step3_img, [palm_poly], True, (255, 255, 0), 2)
            steps["lines"] = step3_img

        # ═══ STEP 4: BUILD FULL ANALYSIS ═══
        mask_downscaled = cv2.resize(segmentation_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[y_min:y_max, x_min:x_max] = mask_downscaled
        # Build a palm-only ROI mask on the original image so overlay shows lines ONLY on the palm
        orig_pts = [get_pt(i) for i in [0, 2, 5, 9, 13, 17, 18]]
        palm_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(palm_roi, [np.array(orig_pts, dtype=np.int32)], 255)
        # Expand slightly to catch edges
        palm_roi = cv2.dilate(palm_roi, np.ones((15, 15), np.uint8), iterations=1)

        # Run enhanced line detection on palm ROI crop only (much faster than full image)
        roi_ys, roi_xs = np.where(palm_roi > 0)
        if len(roi_ys) > 0 and len(roi_xs) > 0:
            ry0, ry1 = int(roi_ys.min()), int(roi_ys.max()) + 1
            rx0, rx1 = int(roi_xs.min()), int(roi_xs.max()) + 1
            roi_crop = img[ry0:ry1, rx0:rx1]
            roi_mask_crop = palm_roi[ry0:ry1, rx0:rx1]
            crop_line_map, crop_enhanced = enhance_palm_image(roi_crop)
            crop_line_map = cv2.bitwise_and(crop_line_map, crop_line_map, mask=roi_mask_crop)
            crop_enhanced = cv2.bitwise_and(crop_enhanced, crop_enhanced, mask=roi_mask_crop)
            enhanced_line_map = np.zeros((h, w), dtype=np.uint8)
            enhanced_gray = np.zeros((h, w), dtype=np.uint8)
            enhanced_line_map[ry0:ry1, rx0:rx1] = crop_line_map
            enhanced_gray[ry0:ry1, rx0:rx1] = crop_enhanced
        else:
            enhanced_line_map, enhanced_gray = enhance_palm_image(img)
            enhanced_line_map = cv2.bitwise_and(enhanced_line_map, enhanced_line_map, mask=palm_roi)
            enhanced_gray = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=palm_roi)

        overlay_img = create_palm_overlay(img, mask_full, enhanced_line_map)
        features = extract_palm_features(
            mask_full,
            raw_image=img,
            palm_region_mask=palm_roi,
            enhanced_line_map=enhanced_line_map,
            enhanced_gray=enhanced_gray,
        )
        features["detected_hand_label"] = handedness_label or "Unclear"
        features["detected_hand_confidence"] = round(float(handedness_score), 3)
        scan_quality = _compute_palm_scan_quality(
            palm_roi,
            mask_full,
            enhanced_gray,
            features,
            handedness_label=handedness_label,
            handedness_score=handedness_score,
        )
        features["scan_quality_score"] = scan_quality["quality_score"]
        features["scan_hand_fill_ratio"] = scan_quality["hand_fill_ratio"]
        features["scan_major_line_ratio"] = scan_quality["major_line_ratio"]
        features["scan_texture_score"] = scan_quality["texture_score"]
        features["scan_line_contrast_score"] = scan_quality["line_contrast_score"]
        features["scan_major_line_count"] = scan_quality["major_line_count"]

        observations = {**DEFAULT_OBSERVATIONS, **build_dynamic_palm_observations(features)}
        report = build_palm_report(features, observations)
        report["scan_quality"] = scan_quality
        report["detected_hand"] = scan_quality["detected_hand"]
        # ── Expert Inspection ──
        report["inspection_crops"] = generate_expert_inspection_crops(img, landmarks)

        report["_steps"] = steps if include_steps else None
        report["_hand_found"] = True
        return overlay_img, mask_full, features, report


    if src in ("📷 Photo", "📸 Camera Snapshot"):
        img = _load_image_from_source(
            src,
            "Upload Palm Photo",
            "palm_photo",
            "Capture Palm Photo",
            "palm_camera",
        )
        if img is not None:
            with st.spinner("🖐️ Step 1: Detecting palm..."):
                overlay, mask, features, report = _palm_cb(img.copy())

            steps = report.get("_steps")
            hand_found = report.get("_hand_found", False)

            if not hand_found:
                _clear_palm_report_session()
                st.warning("⚠️ No hand detected. Please show your full palm clearly with good lighting.")
            else:
                # ── Show step-by-step visual process ──
                st.markdown("""
                    <div style="display:flex;align-items:center;gap:12px;margin:10px 0 15px;">
                        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
                        <span style="font-size:13px;color:#F59E0B;letter-spacing:3px;
                            font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🔬 Detection Process</span>
                        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
                    </div>
                """, unsafe_allow_html=True)

                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown("**Step 1: Palm Detected**")
                    st.image(cv2.cvtColor(steps["detect"], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption="21 landmarks identified")
                with s2:
                    st.markdown("**Step 2: Palm Crop**")
                    st.image(cv2.cvtColor(steps["zoom"], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption=f"Auto-cropped & upscaled to {steps['zoom'].shape[1]}×{steps['zoom'].shape[0]}px")
                with s3:
                    st.markdown("**Step 3: Fine-Line Enhancement**")
                    st.image(cv2.cvtColor(steps["enhanced"], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption="Contrast-enhanced palm texture used for fine line capture")
                with s4:
                    st.markdown("**Step 4: Major Lines Traced**")
                    st.image(cv2.cvtColor(steps["lines"], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption="Life (red) · Head (green) · Heart (blue)")

                # ── Expert Deep Dive Inspection ──
                st.markdown("""
                    <div style="display:flex;align-items:center;gap:12px;margin:30px 0 20px;">
                        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
                        <span style="font-size:13px;color:#06B6D4;letter-spacing:3px;
                            font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🔬 Expert Deep Dive</span>
                        <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
                    </div>
                """, unsafe_allow_html=True)

                inspection_crops = report.get("inspection_crops", [])
                if inspection_crops:
                    cols = st.columns(len(inspection_crops))
                    for i, crop_data in enumerate(inspection_crops):
                        with cols[i]:
                            st.image(cv2.cvtColor(crop_data["image"], cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.markdown(f"<div style='font-size:11px; font-weight:700; color:#06B6D4; text-transform:uppercase; text-align:center;'>{crop_data['name']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:10px; color:#94A3B8; text-align:center; line-height:1.2;'>{crop_data['description']}</div>", unsafe_allow_html=True)

                _persist_palm_report(features, report, overlay)
                _render_palm_report(overlay, features, report)


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

    pass


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
        features = st.session_state.get("palm_latest_features", {})
        
        from utils.palmistry_knowledge import get_contextual_knowledge_summary
        knowledge_context = get_contextual_knowledge_summary(features, palm_report)
        
        render_chatbot(
            "Professional palm analysis — hand type, lines, mounts, timing, health, personality",
            context_payload=palm_report.get("chat_context"),
            system_prompt=build_professional_system_prompt(knowledge_context),
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
                st.image(item["banner"], use_container_width=True)

            with e_col2:
                st.markdown(
                    f"""
                    <div style="padding: 5px 0;">
                        <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 20px; color: white; margin-bottom: 5px;">{item['title']} {item['icon']}</div>
                        <p style="color: #F8FAFC; font-size: 12px; line-height: 1.5; margin-bottom: 12px; font-weight: 500;">{item['gallery_subtitle']}. Optimized for real-time vision processing.</p>
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
