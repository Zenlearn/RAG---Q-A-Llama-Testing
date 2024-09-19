# app.py

# Suppress FutureWarning from transformers (optional)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import necessary libraries
import os
import logging
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List, Mapping, Any

# Load environment variables from a .env file if present (optional)
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app instance
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes. Adjust as needed for security.

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'pdf')
DB_FOLDER = os.path.join(BASE_DIR, 'db')

# Ensure directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Define Together API configurations
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    logger.error("TOGETHER_API_KEY environment variable not set.")
    raise ValueError("TOGETHER_API_KEY environment variable not set.")

# Define a custom LLM class for Together API
class TogetherAPI(BaseLLM):
    api_url: str
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    @property
    def _llm_type(self):
        return "together_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                return answer
            else:
                logger.error(f"Together API Error: {response.status_code} - {response.text}")
                raise ValueError(f"Together API Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error calling Together API: {e}")
            raise

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            answer = self._call(prompt, stop=stop)
            generations.append([Generation(text=answer)])
        return LLMResult(generations=generations)

    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "api_url": self.api_url,
            "model": self.model
        }

    @property
    def _max_tokens(self) -> int:
        return 512

# Initialize the HuggingFace embedding model
try:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("HuggingFaceEmbeddings initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    raise

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

# Initialize the TogetherAPI LLM
try:
    together_llm = TogetherAPI(api_url=TOGETHER_API_URL, api_key=TOGETHER_API_KEY)
    logger.info("TogetherAPI LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize TogetherAPI LLM: {e}")
    raise

# Route to serve the chatbot UI
@app.route("/")
def index():
    return render_template('index.html')

# Endpoint to handle general AI queries
@app.route("/ai", methods=["POST"])
def ai_post():
    logger.info("Received POST request at /ai endpoint.")
    try:
        json_content = request.json
        query = json_content.get("query")

        if not query:
            logger.warning("Query is missing in the request.")
            return jsonify({"error": "Query is missing"}), 400

        logger.info(f"Received query: {query}")

        # Use the TogetherAPI LLM to get the answer
        answer = together_llm(query)
        logger.info(f"LLM response: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error in /ai endpoint: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

# Endpoint to handle PDF-related queries
@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    logger.info("Received POST request at /ask_pdf endpoint.")
    try:
        json_content = request.json
        query = json_content.get("query")

        if not query:
            logger.warning("Query is missing in the request.")
            return jsonify({"error": "Query is missing"}), 400

        logger.info(f"Received PDF query: {query}")

        # Initialize or load Chroma vector store
        vector_store = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=embedding,
        )
        logger.info("Chroma vector store initialized.")

        # Create a retriever
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,               # Top 20 results
                "score_threshold": 0.1 # Minimum similarity score
            },
        )
        logger.info("Retriever created.")

        # Initialize the RetrievalQA chain with TogetherAPI LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=together_llm,
            chain_type="stuff",
            retriever=retriever,
        )
        logger.info("RetrievalQA chain initialized.")

        # Run the chain with the query
        logger.info("Running QA chain...")
        answer = qa_chain.invoke({"query": query})
        logger.info(f"Answer generated: {answer}")

        return jsonify({"answer": answer})

    except AttributeError as e:
        logger.error(f"AttributeError in /ask_pdf endpoint: {e}", exc_info=True)
        return jsonify({"error": "An attribute error occurred. Please check your input."}), 500
    except TypeError as e:
        logger.error(f"TypeError in /ask_pdf endpoint: {e}", exc_info=True)
        return jsonify({"error": "A type error occurred. Please check your input types."}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /ask_pdf endpoint: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

# Endpoint to upload and process PDFs
@app.route("/pdf", methods=["POST"])
def pdf_post():
    logger.info("Received POST request at /pdf endpoint.")
    try:
        # Check for file in request
        if 'file' not in request.files:
            logger.warning("No file part in the request.")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected for uploading.")
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith('.pdf'):
            logger.warning("Uploaded file is not a PDF.")
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Save the uploaded PDF
        file_name = file.filename
        save_path = os.path.join(PDF_FOLDER, file_name)
        file.save(save_path)
        logger.info(f"Saved PDF: {file_name} at {save_path}")

        # Load and split the PDF
        loader = PDFPlumberLoader(save_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from PDF.")

        # Update metadata with source filename
        for doc in docs:
            doc.metadata['source'] = file_name

        # Split documents into chunks
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")

        # Initialize or load Chroma vector store
        vector_store = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=embedding,
        )
        logger.info("Chroma vector store initialized for adding documents.")

        # Add documents to Chroma
        vector_store.add_documents(chunks)
        # Chroma persists automatically upon adding documents

        logger.info(f"Added documents to Chroma and persisted at {DB_FOLDER}.")

        # Respond with success
        response = {
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks),
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return jsonify({"error": "Failed to process PDF"}), 500

# Function to process existing PDFs
def process_existing_pdfs():
    logger.info("Processing existing PDFs in the pdf folder.")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    # Initialize or load Chroma vector store
    vector_store = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding,
    )
    logger.info("Chroma vector store initialized.")

    # Retrieve existing document sources
    existing_sources = set()
    try:
        # Use the internal collection to get documents with metadata
        all_documents = vector_store._collection.get(include=["metadatas", "documents"])
        for doc_id, doc_content, metadata in zip(all_documents['ids'], all_documents['documents'], all_documents['metadatas']):
            if metadata and 'source' in metadata:
                existing_sources.add(metadata['source'])
    except Exception as e:
        logger.warning(f"Could not retrieve existing documents from vector store: {e}")

    for file_name in pdf_files:
        if file_name in existing_sources:
            logger.info(f"Skipping already processed PDF: {file_name}")
            continue

        try:
            save_path = os.path.join(PDF_FOLDER, file_name)
            logger.info(f"Processing PDF: {file_name}")

            # Load and split the PDF
            loader = PDFPlumberLoader(save_path)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from PDF.")

            # Update metadata with source filename
            for doc in docs:
                doc.metadata['source'] = file_name

            # Split documents into chunks
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks.")

            # Add documents to Chroma
            vector_store.add_documents(chunks)
            # Chroma persists automatically upon adding documents
            logger.info(f"Added documents to Chroma and persisted at {DB_FOLDER}.")

        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {e}")

# Function to start the Flask application
def start_app():
    try:
        # Process existing PDFs on startup
        process_existing_pdfs()
        logger.info("Starting Flask server on 0.0.0.0:8080")
        app.run(host="0.0.0.0", port=8080, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        raise

# Entry point
if __name__ == "__main__":
    start_app()