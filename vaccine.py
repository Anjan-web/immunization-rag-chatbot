import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import ingest

load_dotenv()

st.set_page_config(page_title="Immunization Guidelines Chatbot (Groq RAG)", page_icon="📘", layout="wide")

st.title("📘 Immunization Guidelines Chatbot (Groq RAG)")

# Automatically create FAISS index if not present
if not Path("vectorstore_immunization/faiss_index").exists():
    st.warning("No vector index found. Running ingest.py automatically...")
    ingest.main()

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore_immunization/faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"⚠️ Error loading vector store: {e}")
    st.stop()

retriever = db.as_retriever()
model = ChatGroq(model="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

prompt_template = """You are an AI assistant specialized in Immunization guidelines.
Use the following context to answer concisely.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

question = st.text_input("💬 Ask a question about Immunization Guidelines:")
if question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(question)
        st.success(response)
