"""Check if PDF has images (scanned book) and extract some sample images."""
import PyPDF2
import os

pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Palm reading.pdf")

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    # Check a few pages for images
    for i in range(min(5, total_pages)):
        page = reader.pages[i]
        text = page.extract_text()
        print(f"\nPage {i+1}:")
        print(f"  Text length: {len(text) if text else 0}")
        print(f"  Has images: {len(page.images) if hasattr(page, 'images') else 'unknown'}")
        if text:
            print(f"  Text preview: {text[:200]}")
        
        # Check for XObject (images)
        resources = page.get('/Resources')
        if resources:
            xobjects = resources.get('/XObject')
            if xobjects:
                print(f"  XObjects found: {len(xobjects)}")
