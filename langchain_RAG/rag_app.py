import os
import tempfile
import pickle
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv
load_dotenv()
# my_api_key = "AIzaSyBGdG7aYcClJvMQSdQCwQx3d7X_PACPNNA"
# 🔹 Configure Gemini API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Set your Gemini API key

# 🔹 FAISS Index Initialization
D = 768  # Gemini embedding dimension (depends on the model)
faiss_index = faiss.IndexFlatL2(D)  # Euclidean Distance metric
metadata_store = {}  # Store metadata (text, document ID)

# 🔹 System Prompt for LLM
system_prompt = """
You are an AI assistant providing detailed answers based solely on the given context.
Follow these rules:
1. Analyze the provided context carefully.
2. Structure your response logically.
3. If insufficient data is found, state it clearly.
4. Use numbered lists, headings, and proper formatting for clarity.
"""

# 🔹 Function to Generate Gemini Embeddings
def gemini_embed(text: str) -> list[float]:
    """Generates text embeddings using Google's Gemini model."""
    model = "models/embedding-001"  # Embedding model
    response = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    # return response["embedding"]
    if "embedding" in response:
        return response["embedding"]
    else:
        raise ValueError("Failed to generate embeddings.")

# 🔹 Function to Process Uploaded PDF
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file into text chunks."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", ",", "?", "!", " ", ""]
    )
    return text_splitter.split_documents(docs)

# 🔹 Function to Add Data to FAISS
def add_to_vector_store(all_splits: list[Document], file_name: str):
    """Stores document embeddings in FAISS."""
    global faiss_index, metadata_store
    embeddings, texts = [], []

    for idx, split in enumerate(all_splits):
        text = split.page_content
        embedding = gemini_embed(text)
        embeddings.append(embedding)
        texts.append(text)
        metadata_store[len(metadata_store)] = {"text": text, "file_name": file_name}

    # Convert embeddings to NumPy array and add to FAISS
    embeddings_np = np.array(embeddings).astype("float32")
    faiss_index.add(embeddings_np)
    st.success("✅ Data added to FAISS vector store!")

# 🔹 Function to Query FAISS
def query_vector_store(query: str, top_k: int = 5):
    """Retrieves relevant documents from FAISS."""
    query_embedding = np.array([gemini_embed(query)]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx in metadata_store:
            results.append({
                "text": metadata_store[idx]["text"],
                "distance": distances[0][i],
                "file_name": metadata_store[idx]["file_name"]
            })
    return results

# 🔹 Function to Re-rank Results using Cross-Encoder
def re_rank_cross_encoder(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks retrieved documents using a cross-encoder model."""
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    for rank in ranks:
        relevant_text += documents[rank['corpus_id']]
        relevant_text_ids.append(rank['corpus_id'])

    return relevant_text, relevant_text_ids

# 🔹 Function to Call Gemini for Answer Generation
def call_gemini(context: str, prompt: str):
    """Generates responses using Gemini."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Context: {context}\nQuestion: {prompt}")
    return response.text

# 🔹 Streamlit App UI
st.set_page_config(page_title="📖 RAG with FAISS & Gemini")
st.sidebar.header("📌 Upload Documents")

# 📂 Upload PDF File
uploaded_file = st.sidebar.file_uploader("Upload a PDF for Q&A", type=["pdf"])

# Process Uploaded File
if uploaded_file and st.sidebar.button("📌 Process"):
    file_name = uploaded_file.name.replace(" ", "_")
    all_splits = process_document(uploaded_file)
    add_to_vector_store(all_splits, file_name)

# 🔍 Question Answering Section
st.header("🗣️ Ask a Question")
prompt = st.text_area("🔍 Type your question:")

# ask = st.button("🔥 Ask")

if st.button("🔥 Ask"):
    if not prompt:
        st.warning("❗ Please enter a question.")
    else:
        results = query_vector_store(prompt, top_k=5)
        relevant_text = " ".join([res["text"] for res in results])

        # 🔹 Call Gemini for Answer Generation
        response = call_gemini(context=relevant_text, prompt=prompt)
        st.write(response)

        # 📌 Show Relevant Docs
        # with st.expander("📜 See Retrieved Documents"):
        #     st.write(results)
