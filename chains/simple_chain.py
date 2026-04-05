import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate 5 lines about {topic}',
    input_variables=['topic'],
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'IPL'})

print(result)


# Chain Visualisation

chain.get_graph().print_ascii()