import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Doc. Loader
loader = PyPDFLoader('python_book.pdf')

docs = loader.load()


# Prompts
prompt = PromptTemplate(
    template="Write a summary on following text - {text}",
    input_variables=["text"],
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

result = chain.invoke({'text': docs[10].page_content})

# print(len(docs))
# print(f"Summary: {docs[10].page_content}")

print(result)


