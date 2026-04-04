from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

st.header('Research Tools')

user_input = st.text_input('Enter your prompt')

if st.button('Submit'):
    result = model.invoke(user_input)
    st.success(result.content[0]['text'])
