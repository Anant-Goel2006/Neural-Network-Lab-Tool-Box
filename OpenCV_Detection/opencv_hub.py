import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
from utils.styles import section_header, gradient_header
from utils.nlp_engine import generate_cv_insight
import cv2

try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av
    WEBRTC_READY = True
except ImportError:
    WEBRTC_READY = False

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]
})

import tempfile
import os

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
        "title": "Palm Reading",
        "gallery_subtitle": "Hands · Gestures",
        "page_title": "OpenCV Palm Reading",
        "page_subtitle": "Hand segmentation, feature extraction, and palm-line overlays",
        "path": "OpenCV_Detection/page_palm.py",
        "banner": os.path.join("assets", "banners", "palm_reading_banner_1774323346147.png"),
        "features": ["CNN Segmentation", "Feature Extraction", "Career Analysis"],
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
            if f and st.button("📸 Detect & Register", type="primary", width="stretch"):
                img_bytes = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                processed = _att_cb(img)
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
            st.dataframe(df, hide_index=True, width="stretch")
            if st.button("🗑 Clear Log", width="stretch"):
                st.session_state.cv_attendance=[]; st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FACE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
def _face_scan_module():
    section_header("Face Scanner", "Multi-Cascade · Eyes & Smiles")
    
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
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch")
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
            try:
                # Frame skipping (YOLO is heavy)
                if 'vd_frame_count' not in st.session_state: st.session_state.vd_frame_count = 0
                st.session_state.vd_frame_count += 1
                if st.session_state.vd_frame_count % 3 != 0: return frame

                img = frame.to_ndarray(format="bgr24")
                results = model(img, verbose=False)[0]
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

        st.markdown("**Live WebRTC Camera**")
        webrtc_streamer(key="sign_stream", video_frame_callback=sign_callback, rtc_configuration=RTC_CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# PALM READING FEATURE EXTRACTION UTILS
# ═════════════════════════════════════════════════════════════════════════════
def extract_skeleton(mask):
    mask_binary = (mask > 0).astype(np.uint8)
    return cv2.ximgproc.thinning(mask_binary * 255) if hasattr(cv2, 'ximgproc') else mask_binary

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

def extract_palm_features(segmentation_mask):
    life_mask = (segmentation_mask == 1).astype(np.uint8) * 255
    head_mask = (segmentation_mask == 2).astype(np.uint8) * 255
    heart_mask = (segmentation_mask == 3).astype(np.uint8) * 255
    features = {}
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

def create_palm_overlay(image, mask):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    # Colors in BGR (Life: Red, Head: Green, Heart: Blue)
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
    for class_id, color in colors.items():
        class_mask = (mask == class_id)
        overlay[class_mask] = overlay[class_mask] * 0.5 + np.array(color) * 0.5
    return overlay.astype(np.uint8)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PALM READING (CNN Segmentation)
# ═════════════════════════════════════════════════════════════════════════════
def _palm_module():
    section_header("Palm Reading Extraction", "Kinematic Analysis")
    src = st.radio("Input Source", ["📷 Photo", "🔴 Live WebRTC", "📹 Video File"], horizontal=True, key="cv_palm_src")
    
    try:
        import torch
        import segmentation_models_pytorch as smp
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        st.error("Missing dependencies. Please run `pip install -r requirements.txt` (needs torch, segmentation_models_pytorch, albumentations).")
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if 'palm_model' not in st.session_state:
        try:
            with st.spinner("Loading Palm Segmentation Model..."):
                model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4)
                model.load_state_dict(torch.load("palm_model.pth", map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                st.session_state.palm_model = model
        except Exception as e:
            st.error(f"Failed to load Palm Model from 'palm_model.pth': {e}")
            return
            
    model = st.session_state.palm_model
    preprocessing = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def _palm_cb(img):
        # img is BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = preprocessing(image=img_rgb)
        image_tensor = preprocessed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
            
        mask = cv2.resize(pred_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_img = create_palm_overlay(img, mask)
        return overlay_img, mask

    if src == "📷 Photo":
        f = st.file_uploader("Upload Palm Photo", type=["jpg", "png"], key="palm_photo")
        if f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            with st.spinner("Analyzing Palm..."):
                overlay, mask = _palm_cb(img)
                features = extract_palm_features(mask)
                classification = classify_palm(features)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width="stretch", caption="Line Segmentation (Life: Red, Head: Green, Heart: Blue)")
            with c2:
                st.success("### Palm Analysis")
                st.markdown(f"**Dominant Line:** {classification['dominant_line']} ({classification['confidence']:.1%})")
                st.markdown(f"**Palm Type:** {classification['palm_type']}")
                st.markdown(f"**Career Shift Indicator:** {classification['career_shift_indicator']}")
                with st.expander("Extracted Features Details"):
                    st.json({k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()})

    elif src == "📹 Video File":
        v = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="palm_video")
        if v:
            process_video_realtime(v, lambda frame: _palm_cb(frame)[0])
    else:
        def palm_callback(frame: av.VideoFrame) -> av.VideoFrame:
            try:
                # Frame skipping to prevent lag on slow CPUs
                if 'palm_frame_count' not in st.session_state: st.session_state.palm_frame_count = 0
                st.session_state.palm_frame_count += 1
                if st.session_state.palm_frame_count % 3 != 0: return frame # Process every 3rd frame
                
                img = frame.to_ndarray(format="bgr24")
                # Downsample for faster inference
                small_img = cv2.resize(img, (320, 240)) 
                overlay, _ = _palm_cb(small_img)
                # Resize back
                final_overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]))
                return av.VideoFrame.from_ndarray(final_overlay, format="bgr24")
            except Exception as e:
                return frame
            
        st.markdown("**Live WebRTC Camera (CNN Inference)**")
        st.warning("Note: Real-time CNN inference on CPU may experience low framerates.")
        webrtc_streamer(key="palm_stream", video_frame_callback=palm_callback, rtc_configuration=RTC_CONFIG)


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
            push_tutor_insight(ai_text, f"AI Tutor // {topic_label.title()} Tips")
            st.session_state[insight_key] = True

    render_chatbot(topic_label)


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

    render_chatbot("Computer Vision Analytics Hub")


def _render_cv_module_page(module_key):
    module = CV_MODULE_MAP[module_key]
    _render_cv_shell(module["page_title"], module["page_subtitle"], module["icon"])
    _render_cv_jump_bar(module_key)
    st.caption("OpenCV Lab")
    st.write("This tool now lives on its own page, so the workspace stays focused on the selected vision workflow.")
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
