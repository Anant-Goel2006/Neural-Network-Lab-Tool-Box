"""Extract sample images from the PDF to identify the book."""
import PyPDF2
from PIL import Image
import io
import os

pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Palm reading.pdf")
output_dir = os.path.join(os.path.dirname(__file__), "pdf_pages")
os.makedirs(output_dir, exist_ok=True)

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    
    # Extract images from first few pages and some middle/chapter pages
    pages_to_extract = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    
    for page_num in pages_to_extract:
        if page_num >= len(reader.pages):
            continue
        page = reader.pages[page_num]
        
        for img_idx, image in enumerate(page.images):
            img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
            with open(img_path, "wb") as img_file:
                img_file.write(image.data)
            print(f"Saved page {page_num+1} image to {img_path}")
            break  # Only first image per page
