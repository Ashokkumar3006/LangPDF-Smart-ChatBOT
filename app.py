import os
import zipfile
import shutil
from flask import Flask, request, render_template, jsonify, send_from_directory
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

from process.pdf_process import process_and_store

# Load environment variables
load_dotenv()

# Application setup
app = Flask(__name__)

# Directory configurations
PDF_FOLDER = "./pdfs"
TEMP_FOLDER = "./temp_uploads"
CHROMA_DB_PATH = "./chroma"
ALLOWED_EXTENSIONS = {'pdf', 'zip'}

# Ensure directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize embeddings and vector store
embedding_model = OpenAIEmbeddings()

# Load or create Chroma database
if os.path.exists(CHROMA_DB_PATH):
    chroma = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    print("Loaded existing Chroma database.")
else:
    chroma = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    process_and_store(PDF_FOLDER, chroma)
    print("Processed and stored new data in Chroma database.")

# Chat model and memory setup
chat_model = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(input_key="question", memory_key="chat_history")

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an AI assistant. Use the context provided to answer questions clearly and concisely. Include references to sources.\n\n"
        "Context:\n{context}\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Question: {question}\nAnswer:"
    )
])

# LLM Chain
chain = LLMChain(llm=chat_model, prompt=prompt_template, memory=memory)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdfs_from_zip(zip_path):
    """Extract PDF files from a ZIP archive to the upload folder."""
    extracted_pdfs = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.pdf'):
                filename = secure_filename(file_info.filename)
                source = zip_ref.open(file_info)
                target_path = os.path.join(PDF_FOLDER, filename)
                
                with open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                extracted_pdfs.append(filename)
    
    return extracted_pdfs

@app.route('/')
def index():
    return render_template('upload.html')
@app.route('/upload-files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"message": "No selected files"}), 400
    
    try:
        # Collect file paths instead of processing immediately
        file_paths = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(PDF_FOLDER, filename)
                    file.save(file_path)
                    file_paths.append(file_path)
                
                elif filename.lower().endswith('.zip'):
                    zip_path = os.path.join(TEMP_FOLDER, filename)
                    file.save(zip_path)
                    
                    try:
                        # Extract PDFs from ZIP
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
                            
                            for pdf_file in pdf_files:
                                zip_ref.extract(pdf_file, TEMP_FOLDER)
                                extracted_path = os.path.join(TEMP_FOLDER, pdf_file)
                                new_path = os.path.join(PDF_FOLDER, secure_filename(pdf_file))
                                shutil.move(extracted_path, new_path)
                                file_paths.append(new_path)
                    except Exception as e:
                        return jsonify({"message": f"Error processing ZIP: {str(e)}"}), 400
                    finally:
                        # Clean up temp zip file
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
        
        # Asynchronously process files
        from threading import Thread
        
        def process_files():
            try:
                # Use a separate function to process files
                process_and_store(PDF_FOLDER, chroma)
            except Exception as e:
                print(f"Error processing files: {e}")
        
        # Start processing in a separate thread
        Thread(target=process_files).start()
        
        return jsonify({
            "message": "Files uploaded successfully", 
            "files": [os.path.basename(f) for f in file_paths]
        }), 200
    
    except Exception as e:
        return jsonify({"message": f"Upload error: {str(e)}"}), 500
@app.route('/chatbot')
def chat_page():
    return render_template('index.html')
@app.route("/static/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    results = retriever.get_relevant_documents(question)

    question_embedding = embedding_model.embed_query(question)

    sources = []
    for doc in results:
        doc_embedding = embedding_model.embed_query(doc.page_content)
        similarity_score = cosine_similarity([question_embedding], [doc_embedding])[0][0]

        pdf_name = doc.metadata['pdf_name']
        page_number = doc.metadata['page_number']
        file_url = f"/static/pdfs/{pdf_name}"

        sources.append({
            "text": f"Source: {pdf_name} (Page {page_number})",
            "score": f"{similarity_score:.4f}",
            "url": file_url
        })

    context = "\n\n".join([doc.page_content for doc in results])

    inputs = {"question": question, "context": context}
    response = chain.invoke(inputs)
    answer = response["text"]

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True)