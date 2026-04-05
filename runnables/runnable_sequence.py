import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence



load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Prompts
prompt1 = PromptTemplate(
    template="Write a hindi funny joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"],
)

# Parser
parser = StrOutputParser()

# Chain
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

res = chain.invoke({'topic': 'Santa-Banta'})

print(res)