from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the embedding model with 384-dimensional output
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF documents using PyPDFLoader
loader = DirectoryLoader(
    path="data",
    glob="./*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

# Create the Chroma vector store
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized")
