from langchain.vectorstores import Chroma
from process.chunks import chunk_text
from process.extraction import extract_text_from_pdfs


def process_and_store(pdf_folder, chroma):
    pdf_data = extract_text_from_pdfs(pdf_folder)
    for item in pdf_data:
        chunks = chunk_text(item["text"])
        metadata = {"pdf_name": item["pdf_name"], "page_number": item["page_number"]}

        chroma.add_texts(chunks, metadatas=[metadata] * len(chunks))

    chroma.persist()