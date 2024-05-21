import streamlit as st
from langchain.llms import OpenAI
st.set_page_config(page_title = "🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

import os
os.environ["OPENAI_API_KEY"] = "sk-"

def generate_response(input_text):
    llm = OpenAI(model_name = 'gpt-4', temperature = 0)
    st.info(llm(input_text))

with st.form('Question'):
    text = st.text_area('질문 입력: ', 'What types of text models does OpneAI provide?')
    submittied = st.form_submit_button('보내기')
    generate_response(text)