import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import ingest

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Immunization Guidelines Chatbot (Groq RAG)", page_icon="💉", layout="wide")
st.title("💉 Immunization Guidelines Chatbot (Groq RAG)")

# Check if FAISS index exists, otherwise create it
VECTOR_DIR = Path("vectorstore_immunization/faiss_index")
if not VECTOR_DIR.exists():
    st.warning("⚠️ No vector index found. Running ingest.py automatically...")
    try:
        ingest.main()
    except Exception as e:
        st.error(f"Error running ingest.py: {e}")
        st.stop()

# Initialize embeddings and load FAISS vectorstore
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"❌ Failed to load FAISS vectorstore: {e}")
    st.stop()

retriever = db.as_retriever()

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY. Please add it in your Streamlit Secrets or .env file.")
    st.stop()

model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are an expert in Indian Immunization Guidelines. 
Answer the question clearly and concisely using the provided context.
If the answer isn't in the context, say "I'm not sure based on available information."

<context>
{context}
</context>

Question: {input}
""")

# Create chains
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit input
user_query = st.text_input("💬 Ask a question about Immunization Guidelines:")
if user_query:
    with st.spinner("Thinking..."):
        try:
            response = retrieval_chain.invoke({"input": user_query})
            st.success(response["answer"])
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")
