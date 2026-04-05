import os

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

chat_prompt_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistance'),
    ('human', 'Tell me about the {topic}'),
])
prompt = chat_prompt_template.invoke({'domain': 'GenAI', 'topic': 'RAG'})

response = model.invoke(prompt)

print(response.content)
