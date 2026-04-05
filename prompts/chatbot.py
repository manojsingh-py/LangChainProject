import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st



load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Chat History
chat_history = []


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot 🤖")


# User input field
user_input = st.text_input("Enter your message", key="user_input")

# Process input
if user_input:
    st.session_state.messages.append(("You", user_input))
    chat_history.append(user_input)


    response = model.invoke(chat_history)
    chat_history.append(response.content)
    st.session_state.messages.append(("AI", response.content))

# Display chat history
for role, msg in st.session_state.messages:

    if role == "You":
        # st.markdown(
        #     f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:5px'>🧑 {msg}</div>",
        #     unsafe_allow_html=True
        # )
        st.info(f"**{role}:** {msg}")
    elif role == "AI":
        # st.markdown(
        #     f"<div style='background-color:#E8E8E8;padding:10px;border-radius:10px;margin-bottom:5px'>🚀 {msg}</div>",
        #     unsafe_allow_html=True
        # )
        st.success(f"**{role}:** {msg}")
