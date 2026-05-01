import os

with open(r'OpenCV_Detection/opencv_hub.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start = -1
end = -1
for i, line in enumerate(lines):
    if line.startswith('def _render_palm_report('):
        start = i
    if start != -1 and i > start and line.startswith('def '):
        end = i - 1
        break

if start != -1 and end != -1:
    new_func = """def _render_palm_report(overlay, features, report):
    import uuid

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — PALM SCAN (full-width image + key metrics)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('''
        <div style="display:flex;align-items:center;gap:14px;margin:10px 0 20px;">
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
            <span style="font-size:14px;color:#F59E0B;letter-spacing:4px;
                font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🖐️ AI Palm Scan</span>
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption="High-Fidelity AI Line Extraction Overlay",
        )
    with c2:
        dq = report.get("detection_quality", 1.0)
        scan_quality = report.get("scan_quality", {})
        if dq < 0.55:
            render_content_card(
                "⚠️ Scan Quality Low",
                "Better lighting and a flatter palm will improve the reading.",
                accent_color="#F59E0B", icon="⚠️",
            )
        fine_detail = features.get('total_fine_lines', 0)
        render_info_grid([
            ("Hand Type", report.get("hand_type", {}).get("type", "Mixed")),
            ("Detected Hand", report.get("detected_hand", "Unclear")),
            ("Dominant Line", report.get("dominant_line", "Unknown")),
            ("Detection", f"{dq:.0%}"),
            ("Major Lines", f"{scan_quality.get('major_line_count', 0)}/3"),
            ("Fine Lines", str(fine_detail)),
        ])

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2 — FULL AI READING
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('''
        <div style="display:flex;align-items:center;gap:14px;margin:30px 0 20px;">
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
            <span style="font-size:14px;color:#F59E0B;letter-spacing:4px;
                font-weight:600;text-transform:uppercase;font-family:'Inter',sans-serif;">🤖 Full AI Analysis</span>
            <div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>
        </div>
    ''', unsafe_allow_html=True)

    # Render the full AI-generated reading
    if "full_ai_reading" in report:
        st.markdown(report["full_ai_reading"], unsafe_allow_html=True)
    else:
        render_content_card(
            "🔮 AI Reading Summary",
            report.get("summary", "Reading generated offline...").replace('\\n', '<br>'),
            accent_color="#8B5CF6", icon="🔮",
        )
"""
    
    nl = lines[0][-2:] if lines[0].endswith('\r\n') else '\n'
    new_func_lines = [line + nl for line in new_func.split('\n')]
    
    lines[start:end+1] = new_func_lines
    
    with open(r'OpenCV_Detection/opencv_hub.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('SUCCESS')
else:
    print('FAIL')
