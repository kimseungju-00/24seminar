import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

os.environ["OPENAI_API_KEY"] = "sk-"

def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key = 'chat_history', return_messages = True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature = 0, model_name = 'gpt-3.5-turbo-16k-0613'),
        retriever = vectorstore.as_retriever(),
        get_chat_history = lambda h: h,
        memory = memory
    )
    return conversation_chain

user_uploads = st.file_uploader("파일을 업로드해주세요~", accept_multiple_files = True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("처리중.."):
            raw_text = get_pdf_text(user_uploads)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

if user_query := st.chat_input("질문을 입력해주세요~"):
    if 'conversation' in st.session_state:
        result = st.session_state.conversation({
            "question": user_query,
            "chat_history": st.session_state.get('chat_history', [])
        })
        response = result["answer"]
    else:
        response = "먼저 문서를 업로드해주세요~."
    with st.chat_message("assistant"):
        st.write(response)