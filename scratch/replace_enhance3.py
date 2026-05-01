import os

with open(r'OpenCV_Detection/opencv_hub.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start = -1
end = -1
for i, line in enumerate(lines):
    if line.startswith('def enhance_palm_image(image):'):
        start = i
    if start != -1 and i > start and line.startswith('def '):
        end = i - 1
        break

if start != -1 and end != -1:
    new_func = """def enhance_palm_image(image):
    \"\"\"
    Ultra-fast, high-fidelity palm line detector optimized for 'ink-stamp' extraction.
    \"\"\"
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
"""
    
    nl = lines[0][-2:] if lines[0].endswith('\r\n') else '\n'
    new_func_lines = [line + nl for line in new_func.split('\n')[:-1]]
    
    lines[start:end+1] = new_func_lines
    
    with open(r'OpenCV_Detection/opencv_hub.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('SUCCESS')
else:
    print('FAIL')
