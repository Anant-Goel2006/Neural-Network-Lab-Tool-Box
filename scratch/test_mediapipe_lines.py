import cv2
import mediapipe as mp
import numpy as np

def extract_palm_lines_with_mediapipe(img_bgr):
    # Initialize mediapipe hands
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.1
    ) as hands:
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            print("No hands detected.")
            return None, None
            
        h, w = img_bgr.shape[:2]
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Helper to get pixel coords
        def get_pt(idx):
            return np.array([int(landmarks[idx].x * w), int(landmarks[idx].y * h)])

        # Key anatomical points
        wrist = get_pt(0)
        thumb_base = get_pt(2)
        index_base = get_pt(5)
        middle_base = get_pt(9)
        ring_base = get_pt(13)
        pinky_base = get_pt(17)
        pinky_side = get_pt(18) # lower near pinky
        
        # ── PALM BOUNDARY MASK ──
        palm_poly = np.array([wrist, thumb_base, index_base, middle_base, ring_base, pinky_side], dtype=np.int32)
        palm_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(palm_mask, [palm_poly], 255)
        
        # Erode to avoid background
        kernel = np.ones((7, 7), np.uint8)
        palm_mask = cv2.erode(palm_mask, kernel, iterations=1)
        
        # ── EXTRACT CREASES (LINES) ──
        # Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive Threshold (lines are darker than skin)
        inv = cv2.bitwise_not(enhanced) # lines are now brighter
        blur = cv2.GaussianBlur(inv, (5, 5), 0)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        
        # Keep only lines inside palm
        lines_mask = cv2.bitwise_and(edges, edges, mask=palm_mask)
        
        # Remove small noise
        lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        # ── SEGMENT LINES INTO HEART, HEAD, LIFE ──
        # We will split the palm mask into 3 distinct anatomical regions using distance fields.
        # Life line region: bounded by curve from index_base to wrist curving around thumb_base.
        # Heart line region: upper horizontal area below finger bases (index to pinky).
        # Head line region: middle area crossing from index_base towards middle of pinky_side.
        
        segmentation_mask = np.zeros((h, w), dtype=np.uint8)
        
        # We can dynamically assign each pixel inside palm_mask to a region depending on distance to characteristic lines
        # Expected Life Line: Vector from index_base to wrist
        # Expected Heart Line: Vector from index_base to pinky_base
        # Expected Head Line: Vector from index_base to halfway between pinky_side and wrist
        
        # Simple heuristic to split image coordinates
        y_indices, x_indices = np.where(palm_mask > 0)
        pts = np.vstack((x_indices, y_indices)).T
        
        import math
        def pt_to_segment_dist(p, a, b):
            # p: points [N, 2], a, b: line segment endpoints [2]
            ab = b - a
            ap = p - a
            ab_norm = np.linalg.norm(ab)
            if ab_norm == 0: return np.linalg.norm(ap, axis=1)
            t = np.sum(ap * ab, axis=1) / (ab_norm ** 2)
            t = np.clip(t, 0, 1)
            proj = a + t[:, np.newaxis] * ab
            return np.linalg.norm(p - proj, axis=1)
            
        d_life = pt_to_segment_dist(pts, (index_base+wrist)//2, wrist) # Actually life curves around thumb
        d_heart = pt_to_segment_dist(pts, (index_base+middle_base)//2, pinky_side)
        d_head = pt_to_segment_dist(pts, index_base, (pinky_side+wrist)//2)
        
        dists = np.vstack((d_life, d_head, d_heart)).T
        labels = np.argmin(dists, axis=1) + 1 # 1: life, 2: head, 3: heart
        
        for i, (x, y) in enumerate(pts):
            if lines_mask[y, x] > 0:
                segmentation_mask[y, x] = labels[i]
                
        return segmentation_mask, lines_mask
        
if __name__ == "__main__":
    print("Script compiled successfully.")
    # Assuming user uploaded an image or we have one
