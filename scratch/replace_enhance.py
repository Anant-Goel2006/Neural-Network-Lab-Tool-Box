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
    new_lines = [
        'def enhance_palm_image(image):\n',
        '    """\n',
        '    High-fidelity palm line detector optimized for ink-stamp style extraction.\n',
        '    Replaces noisy filters with robust multi-scale adaptive thresholding to perfectly\n',
        '    isolate fine ridges and major lines as bright lines on a dark background.\n',
        '    """\n',
        '    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n',
        '\n',
        '    # 1. CLAHE to equalize lighting across the palm\n',
        '    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\n',
        '    clahe_img = clahe.apply(gray)\n',
        '    \n',
        '    # 2. Denoise slightly to prevent pores from becoming lines\n',
        '    blurred = cv2.medianBlur(clahe_img, 3)\n',
        '\n',
        '    # 3. Multi-scale Adaptive Thresholding (Finds pixels darker than local neighborhood)\n',
        '    # This precisely mimics the handprint effect seen in professional palm readers\n',
        '    \n',
        '    # Fine lines (hairline ridges)\n',
        '    fine = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)\n',
        '    \n',
        '    # Medium lines (secondary lines)\n',
        '    med = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)\n',
        '    \n',
        '    # Broad lines (major creases)\n',
        '    broad = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 4)\n',
        '    \n',
        '    # 4. Combine all line maps\n',
        '    line_map = cv2.bitwise_or(cv2.bitwise_or(fine, med), broad)\n',
        '\n',
        '    # 5. Morphological cleanup — remove noise dots, connect broken segments.\n',
        '    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))\n',
        '    line_map = cv2.morphologyEx(line_map, cv2.MORPH_CLOSE, kernel_close, iterations=1)\n',
        '    \n',
        '    # Remove single-pixel noise\n',
        '    line_map = cv2.medianBlur(line_map, 3)\n',
        '\n',
        '    # 6. Remove very small connected components (noise).\n',
        '    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_map, connectivity=8)\n',
        '    min_area = max(10, int(line_map.shape[0] * line_map.shape[1] * 0.00010))\n',
        '    for i in range(1, num_labels):\n',
        '        if stats[i, cv2.CC_STAT_AREA] < min_area:\n',
        '            line_map[labels == i] = 0\n',
        '\n',
        '    return line_map, clahe_img\n',
        '\n'
    ]
    
    # Fix line endings
    nl = lines[0][-2:] if lines[0].endswith('\r\n') else '\n'
    if nl == '\r\n':
        new_lines = [l.replace('\n', '\r\n') for l in new_lines]
        
    lines[start:end+1] = new_lines
    with open(r'OpenCV_Detection/opencv_hub.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('SUCCESS')
