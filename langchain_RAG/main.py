import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

gemini_api_key = os.getenv('GOOGLE_API_KEY')
# gemini_api_key = "AIzaSyA_bEYm4EvvktM9Xcnjv3tZnE9QJB_rjtE"

class set_RAG:

    def __init__(self):

        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, 
                                                   separator='\n')
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                                       google_api_key=gemini_api_key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                                          convert_system_message_to_human=True)
        self.retrieval_qa_chat_propmt = hub.pull('langchain-ai/retrieval-qa-chat')

        self.combine_docs_chain = create_stuff_documents_chain(
            self.llm,
            self.retrieval_qa_chat_propmt
        )
        self.new_vectorstore = None

    def set_vectors(self,pdf_path,pdf_name):
        """
        _summary_
        Args:
        pdf_name (str) : pass pdf_name 
        pdf_path (str) : pass pdf_path

        Returns:
        vectorstore : It will return the vectorstore
        """
        self.pdf_path, self.pdf_name = pdf_path, pdf_name
        loader = PyPDFLoader(os.path.join(self.pdf_path, self.pdf_name))
        documents = loader.load()

        split_documents = self.text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(split_documents, self.embeddings)

        # Save the vectorstore
        vectorstore.save_local(f"vectorstorages/faiss_index_{self.pdf_name}")

        # load the vectorstore
        self.new_vectorstore = FAISS.load_local(
            f"vectorstorages/faiss_index_{self.pdf_name}", 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # return vectorstore
    
    def get_answer(self, question):
        retrieval_chain = create_retrieval_chain(
            self.new_vectorstore.as_retriever(),
            self.combine_docs_chain
        )

        res = retrieval_chain.invoke({"input":question})

        return res['answer']
    
if __name__ == "__main__":
    pdf_path = "D:/GenAI-Practice/GenAI-Projects/langchain_RAG/docs/"
    pdf_name = "Attention Is All You Need.pdf"
    question = 'What is scaled dot-product attention?'
    # question = input("Enter a question?")
    qna = set_RAG()
    qna.set_vectors(pdf_path, pdf_name)
    answer = qna.get_answer(question)
    print(f"Question {question}")
    print(f"Answer \n{answer}")
