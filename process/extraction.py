import os
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_folder):
    pdf_data = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pdf_data.append({"text": text, "pdf_name": pdf_file, "page_number": page_num + 1})
    return pdf_data