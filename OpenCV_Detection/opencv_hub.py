import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
from utils.styles import section_header, render_nlp_insight, gradient_header
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
    section_header("Face Log & Attendance", "Match Faces · Export CSV")
    render_nlp_insight(generate_cv_insight("attendance"), "Optical NLP // Live Tracking", "#f59e0b")
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
    section_header("Face Scanner", "Multi-Cascade · Eyes & Smiles")
    render_nlp_insight(generate_cv_insight("face"), "Optical NLP // Live Tracking", "#38bdf8")
    
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
    render_nlp_insight(generate_cv_insight("vehicle"), "Optical NLP // Live Tracking", "#f59e0b")
    
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
    render_nlp_insight(generate_cv_insight("sign"), "Optical NLP // Pattern Recognition", "#E11D48")
    
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
    render_nlp_insight(generate_cv_insight("palm"), "Optical NLP // Gesture Parsing", "#a855f7")
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
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Line Segmentation (Life: Red, Head: Green, Heart: Blue)")
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
            img = frame.to_ndarray(format="bgr24")
            overlay, _ = _palm_cb(img)
            return av.VideoFrame.from_ndarray(overlay, format="bgr24")
            
        st.markdown("**Live WebRTC Camera (CNN Inference)**")
        st.warning("Note: Real-time CNN inference on CPU may experience low framerates.")
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
        active = st.session_state.get("cv_module", None) == key
        border = f"{clr}" if active else "#000"
        cols[i].markdown(f"""
        <div style="background:#09090B; border: 2px solid {'#E11D48' if active else '#27272A'};
            padding:16px; text-align:center; box-shadow: 4px 4px 0px {'#E11D48' if active else '#000'};
            transition:all 0.2s; height: 115px; display: flex; flex-direction:column; justify-content:center;
            margin-bottom: 10px;">
            <div style="font-size:32px; margin-bottom:4px; text-shadow: 2px 2px 0px #000;">{icon}</div>
            <div style="font-family:'Oswald', sans-serif; font-size:18px; color:#FAFAFA; font-weight:500; text-transform: uppercase;">{title}</div>
            <div style="font-size:12px; font-family:'Inter'; font-weight:600; color:#71717A; margin-top:4px; text-transform: uppercase;">{desc}</div>
        </div>""", unsafe_allow_html=True)
        if cols[i].button(f"Open {title}", key=f"cv_btn_{key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.cv_module=key; st.rerun()

    mod = st.session_state.get("cv_module", None)
    st.divider()

    if mod == "attendance": _attendance_module()
    elif mod == "face_scan": _face_scan_module()
    elif mod == "vehicle": _vehicle_module()
    elif mod == "sign": _sign_module()
    elif mod == "palm": _palm_module()
    else:
        st.markdown("""
        <div style="background: rgba(10, 10, 20, 0.7); backdrop-filter: blur(16px); border: 1px solid rgba(139,92,246,0.3); border-radius: 20px; padding: 60px 30px; text-align: center; margin-top: 20px; box-shadow: 0 16px 40px rgba(0,0,0,0.6); word-wrap: break-word; overflow-wrap: break-word;">
            <div style="font-size: 72px; margin-bottom: 20px; filter: drop-shadow(0 0 20px rgba(0, 240, 255, 0.4));">🛰️</div>
            <h2 style="font-family: 'Oswald', sans-serif; font-size: 42px; color: #FAFAFA; margin: 0; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">NEUROLAB Analytics Suite</h2>
            <div style="width: 80px; height: 3px; background: linear-gradient(90deg, transparent, #00f0ff, transparent); margin: 20px auto;"></div>
            <p style="font-family: 'Inter', sans-serif; font-weight: 400; color: #00f0ff; font-size: 16px; letter-spacing: 2px;">
                AWAITING OPTICAL UPLINK OVERRIDE
            </p>
        </div>
        """, unsafe_allow_html=True)
