import os

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage('You are a helpful assistance'),
    HumanMessage('Tell me about the RAG')
]

response = model.invoke(messages)
messages.append(AIMessage(response.content))

print(response.content)
