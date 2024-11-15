# extract.py
import os
import fitz  # PyMuPDF
from langchain.docstore.document import Document

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []

    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text("text")
        images = page.get_images(full=True)
        image_filenames = []

        # Extract images
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"static/images/image_page{page_number}_{img_index}.{img_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(img_bytes)
            image_filenames.append(img_filename)

        # Create a LangChain Document
        metadata = {
            'page_number': page_number,
            'image_filenames': image_filenames
        }
        langchain_doc = Document(page_content=text, metadata=metadata)
        documents.append(langchain_doc)

    return documents