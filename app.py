# app.py

import os
import logging
from logging.handlers import RotatingFileHandler
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.document_loaders import PDFPlumberLoader  # Ensure this is up-to-date
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List, Mapping, Any
from dotenv import load_dotenv
import warnings

# Suppress specific FutureWarning from transformers (Optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Configuration and Setup
load_dotenv()

# Logging Configuration
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
log_handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=2)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

# Console Logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

logger.info("Starting application...")

# Flask App
app = Flask(__name__)

# CORS Configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://13.200.228.207:8080")
CORS(app, resources={r"/*": {"origins": [FRONTEND_ORIGIN]}})

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'pdf')
DB_FOLDER = os.path.join(BASE_DIR, 'db')

# Ensure directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Ollama API Configuration
LLAMA_API_URL = os.getenv("LLAMA_API_URL")  # e.g., "http://localhost:11434/v1/models/llama-3.1/completions"
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")  # If required

if not LLAMA_API_URL:
    logger.error("LLAMA_API_URL environment variable not set.")
    raise ValueError("LLAMA_API_URL environment variable not set.")
else:
    logger.info("LLAMA_API_URL loaded successfully.")

# Ollama LLM Class with Enhanced Logging and Error Handling
class OllamaLLM(BaseLLM):
    api_url: str
    api_key: Optional[str] = None
    model: str = "llama-3.1"

    @property
    def _llm_type(self):
        return "ollama_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "stop": stop if stop else ["\n"]
        }

        logger.debug(f"Sending payload to Ollama API: {payload}")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            logger.debug(f"Received response: {response.status_code} - {response.text}")
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Full API response: {result}")

            choices = result.get('choices', [])
            if not choices:
                logger.error("No choices found in API response.")
                return ""

            first_choice = choices[0]
            text = first_choice.get('text')

            if isinstance(text, dict):
                logger.warning(f"Expected 'text' to be a string, but got dict: {text}")
                # Adjust the key based on actual response structure
                text = text.get('content', '')  # Example key; modify as needed
                if not text:
                    logger.error("No 'content' field found in 'text' dict.")
                    return ""

            if not isinstance(text, str):
                logger.error(f"Unexpected type for 'text': {type(text)}. Expected string.")
                return ""

            answer = text.strip()
            logger.debug(f"Extracted answer: {answer}")
            return answer

        except requests.exceptions.Timeout:
            logger.error("Request to Ollama API timed out.")
            raise
        except requests.exceptions.ConnectionError:
            logger.error("Connection error occurred while calling Ollama API.")
            raise
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
            raise
        except KeyError:
            logger.error("Unexpected response structure from Ollama API.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
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

# Initialize Embeddings and Vector Store
try:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("HuggingFaceEmbeddings initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    raise

try:
    vector_store = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding,
    )
    logger.info("Chroma vector store initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Chroma vector store: {e}")
    raise

# Initialize the Ollama LLM
try:
    ollama_llm = OllamaLLM(api_url=LLAMA_API_URL, api_key=LLAMA_API_KEY)
    logger.info("Ollama LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Ollama LLM: {e}")
    raise

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

# Routes and Endpoints
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route("/ai", methods=["POST"])
def ai_post():
    logger.info("Received POST request at /ai endpoint.")
    try:
        json_content = request.get_json()
        if not json_content:
            logger.warning("No JSON content in the request.")
            return jsonify({"error": "Invalid JSON"}), 400

        query = json_content.get("query")

        if not query:
            logger.warning("Query is missing in the request.")
            return jsonify({"error": "Query is missing"}), 400

        logger.info(f"Received query: {query}")

        # Use the Ollama LLM to get the answer
        answer = ollama_llm(query)
        logger.info(f"LLM response: {answer}")

        return jsonify({"answer": answer})

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error in /ai endpoint: {req_err}")
        return jsonify({"error": "External service error"}), 502
    except Exception as e:
        logger.error(f"Error in /ai endpoint: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    logger.info("Received POST request at /ask_pdf endpoint.")
    try:
        json_content = request.get_json()
        if not json_content:
            logger.warning("No JSON content in the request.")
            return jsonify({"error": "Invalid JSON"}), 400

        query = json_content.get("query")

        if not query:
            logger.warning("Query is missing in the request.")
            return jsonify({"error": "Query is missing"}), 400

        logger.info(f"Received PDF query: {query}")

        # Create a retriever
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,               # Top 20 results
                "score_threshold": 0.1 # Minimum similarity score
            },
        )
        logger.debug("Retriever created.")

        # Retrieve relevant documents using the updated invoke method
        try:
            results = retriever.invoke({"query": query})  # Updated to use 'invoke'
            # Depending on the retriever's 'invoke' response structure, adjust accordingly
            # Assuming it returns a list of documents
        except AttributeError as attr_err:
            logger.error(f"Retriever 'invoke' method not found: {attr_err}")
            return jsonify({"error": "Retriever invocation failed"}), 500

        # Check if 'results' is a list
        if not isinstance(results, list):
            logger.error(f"Expected list of documents, got {type(results)} instead.")
            return jsonify({"error": "Unexpected retriever response format"}), 500

        logger.info(f"Retrieved {len(results)} chunks for the query.")

        # Log retrieved chunks for verification
        for i, doc in enumerate(results[:3]):  # Log first 3 retrieved documents
            logger.debug(f"Retrieved Chunk {i+1}: {doc.page_content[:100]}...")

        # Initialize the RetrievalQA chain with Ollama LLM
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ollama_llm,
                chain_type="stuff",
                retriever=retriever,
            )
            logger.debug("RetrievalQA chain initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalQA chain: {e}")
            return jsonify({"error": "Failed to initialize QA chain"}), 500

        # Invoke the QA chain using the updated 'invoke' method
        try:
            qa_result = qa_chain.invoke({"query": query})  # Updated to use 'invoke'
            logger.debug(f"QA Chain result: {qa_result}")
        except Exception as e:
            logger.error(f"Failed to invoke QA chain: {e}")
            return jsonify({"error": "Failed to execute QA chain"}), 500

        # Extract the answer
        if isinstance(qa_result, dict):
            response_text = qa_result.get("result", "No result found.")
        else:
            logger.warning(f"Unexpected type for qa_result: {type(qa_result)}. Expected dict.")
            response_text = str(qa_result)

        logger.info(f"Final Answer: {response_text}")

        response_answer = {"answer": response_text}

        return jsonify(response_answer)

    except Exception as e:
        logger.error(f"Error in /ask_pdf endpoint: {e}")
        return jsonify({"error": "Failed to process PDF query"}), 500

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

        # Log sample chunks for verification
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            logger.debug(f"Chunk {i+1}: {chunk.page_content[:100]}...")

        # Add documents to Chroma
        vector_store.add_documents(chunks)
        logger.info(f"Added documents to Chroma and persisted at {DB_FOLDER}.")

        # Respond with success
        response = {
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks),
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return jsonify({"error": "Failed to process PDF"}), 500

# Processing Existing PDFs on Startup
def process_existing_pdfs():
    logger.info("Processing existing PDFs in the pdf folder.")
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDFs to process.")
    except Exception as e:
        logger.error(f"Failed to list PDFs in {PDF_FOLDER}: {e}")
        return

    if not pdf_files:
        logger.info("No existing PDFs to process.")
        return

    # Retrieve existing document sources to avoid reprocessing
    existing_sources = set()
    try:
        # Access the underlying chromadb collection
        collection = vector_store._collection  # Note: Accessing a protected member
        all_documents = collection.get(include=["metadatas", "documents"])

        for metadata in all_documents['metadatas']:
            if metadata and 'source' in metadata:
                existing_sources.add(metadata['source'])
        logger.info(f"Existing sources in Chroma: {existing_sources}")
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

            # Log sample chunks for verification
            for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                logger.debug(f"Chunk {i+1}: {chunk.page_content[:100]}...")

            # Add documents to Chroma
            vector_store.add_documents(chunks)
            logger.info(f"Added documents to Chroma and persisted at {DB_FOLDER}.")

        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {e}")

# Application Entry Point
def start_app():
    try:
        # Process existing PDFs on startup
        process_existing_pdfs()
        logger.info("Starting Flask server on 0.0.0.0:8080")
        app.run(host="0.0.0.0", port=8080, debug=False)  # Set debug=False for production
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        raise

# Entry point
if __name__ == "__main__":
    start_app()
