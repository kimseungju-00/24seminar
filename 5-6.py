import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_KEY"] = "sk-"

langs = ["Korean", "Japanese", "chinese", "English"]
left_co, cent_co, last_co = st.columns(3)

with st.sidebar:
    language = st.radio("번역을 원하는 언어를 선택해주세요.:", langs)

st.markdown('### 언어 번역 서비스예요~')
prompt = st.text_input("번역을 원하는 텍스트를 입력하세요")

trans_template = PromptTemplate(
    input_variables = ['trans'],
    template = 'Your task is to translate this text to ' + language + 'TEXT: {trans}'
)

memory = ConversationBufferMemory(input_key = 'trans', memory_key = 'chat_history')
llm = OpenAI(model_name = 'gpt-4', temperature = 0)
trans_chain = LLMChain(llm = llm, prompt = trans_template, verbose = True,
                       output_key = 'translate', memory = memory)

if st.button("번역"):
    if prompt:
        response = trans_chain({'trans':prompt})
        st.markdown(response['translate'])
