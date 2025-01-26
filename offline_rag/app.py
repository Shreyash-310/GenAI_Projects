
import os 
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder

from streamlit.runtime.uploaded_file_manager import UploadedFile

import ollama

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """
    Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile('wb', suffix='.pdf', delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    # os.unlink(temp_file.name)   # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n","\n",",","?","!"," ",""]
    )

    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    """Gets or create a ChromaDB collection for vector storage.

    Creates a Ollama Embedding function using nomic-embed-text model and initializes a persistent ChromaDB client.
    Returns a collection that can be used to store and query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
        function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url = "http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")

    return chroma_client.get_or_create_collection(
        name='rag_app',
        embedding_function=ollama_ef,
        metadata={"hnsw:space":"cosine"}
    )

def add_to_vector_collection(all_splits: list[Document], file_name:str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    st.success("Data added to the vector store!")

def query_collection(prompt: str, n_results:int=10):
    """Queries the vector collection with a given propmt to retrieve relevant documents.
    Args:
        prompt: The search query text to find relevant dcouments.
        n_results: maximum number of results to return. Default is 10.
    
    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues quering the collection.
    """

    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context:str, prompt:str):
    """Calls the language model with context and prompt to generate a response.

    Use Ollama to stream responses from a language model by providing context and a question prompt.
    The model uses a system prompt to format andf ground its rsponses appropriately.

    Args:
        context: String containing the relevent context for answering the question.
        prompt : String containing the user's question.

    Yields:
        String chunks of the generated response as they become available from the model.

    Raises:
        OllamaError: If there are issues communicating with ollama API.
    """

    response = ollama.chat(
        model='llama3.2:3b',
        stream=True,
        messages=[
            {
                "role":"system",
                "content":"system_prompt",
            },
            {
                "role":"user",
                "content":f"Context: {context}, Question:{prompt}"
            }
        ]
    )

    for chunk in response:
        if chunk['done'] is False:
            yield chunk['message']['content']
        else:
            break

def re_rank_cross_encoder(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank tyhe input documents based on their
    relevance to the query prompt. Returns the concatenated text of the top 3 most relevant 
    documents along with their indices.

    Args:
        documents: list of documents strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
        - relevant_text (str): Concatenated text from the top 3 ranked documents
        - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If document list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text+= documents[rank['corpus_id']]
        relevant_text_ids.append(rank['corpus_id'])

    return relevant_text, relevant_text_ids

if __name__ == "__main__":

    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        st.header("RAG Question Answer")
        uploaded_file = st.file_uploader(
            "Upload the files for QnA", type=['pdf'], accept_multiple_files=False
        )

        process = st.button(
            "Process"
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-":"_","-":"."," ":"_",})
            )
            all_splits = process_document(uploaded_file)
            # st.write(all_splits)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoder(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See most relevant document ids"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
