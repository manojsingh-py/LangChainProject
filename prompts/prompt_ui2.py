import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_REPO_ID"),
    task=os.getenv("HUGGINGFACE_TASK"),
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

st.title('AI Research Tool')

paper = st.selectbox("Select a Research topics:",
                          ['Multimodal AI', 'Agentic AI & Automation', 'Generative AI & LLMs', 'Computer Vision',
                           'Reinforcement Learning', 'RAG'])
style = st.selectbox("Select a Explanation Style:", ['Beginner-friendly', 'Technical', 'Code-Oriented', 'Mathematical'])

length = st.selectbox("Select a Explanation Length:", ['Small (1-2 Paragraph)', 'Medium (2-5 Paragraph)', 'Long (5-8 Paragraph)'])

template = load_prompt('template.json')

prompt = template.invoke({
        'paper': paper,
        'style': style,
        'length': length,
    }
)

if st.button("Submit"):
    res = model.invoke(prompt)
    st.success(res.content)