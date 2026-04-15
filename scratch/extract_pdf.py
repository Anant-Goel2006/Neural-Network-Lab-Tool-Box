"""Extract text from Palm reading.pdf for training the palm module."""
import PyPDF2
import os

pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Palm reading.pdf")
output_path = os.path.join(os.path.dirname(__file__), "palm_reading_extracted.txt")

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    all_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            all_text.append(f"--- PAGE {i+1} ---\n{text.strip()}")
    
    full_text = "\n\n".join(all_text)
    
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(full_text)
    
    print(f"Extracted {len(all_text)} pages with text")
    print(f"Total characters: {len(full_text)}")
    print(f"Saved to: {output_path}")
    
    # Print first 5000 chars for preview
    print("\n\n=== PREVIEW (first 5000 chars) ===\n")
    print(full_text[:5000])
