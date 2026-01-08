from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "nodejs.pdf"

# Load PDF
loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=40
)
texts = splitter.split_documents(docs)

print(f"Created {len(texts)} chunks")

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Store vectors (IMPORTANT: use `texts`)
vector_store = QdrantVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="my_collection"
)

print("Indexing of documents done!")
