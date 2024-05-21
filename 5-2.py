from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

documents = TextLoader("AI.txt").load()

def split_docs(documents, chunk_size = 1000, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

os.environ["OPENAI_API_KEY"] = "sk-"

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name = model_name)

chain = load_qa_chain(llm, chain_type = "stuff", verbose = True)

query = "AI란?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents = matching_docs, question = query)

answer

