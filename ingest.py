import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectorstore_immunization/faiss_index")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def load_pdfs():
    docs = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    if not docs:
        raise RuntimeError("⚠️ No PDFs found in /data. Please upload guideline PDFs.")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

def main():
    print("📚 Loading PDFs from /data...")
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages.")

    print("🔪 Splitting into chunks...")
    chunks = chunk_docs(docs)
    print(f"Created {len(chunks)} chunks.")

    print("🧠 Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(VECTOR_DIR))
    print("✅ Saved FAISS index to:", VECTOR_DIR)

if __name__ == "__main__":
    main()
