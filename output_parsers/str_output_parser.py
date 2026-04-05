import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# 1. Details report on topic

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2. Summary on the
template2 = PromptTemplate(
    template='Write a 5 line summary on following text {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'Black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result1.content)
print(result2.content)
