import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()


url = r'https://docs.python.org/3/tutorial/datastructures.html'

loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content)



# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Prompts
prompt = PromptTemplate(
    template="Write a summary for following text - {text}",
    input_variables=["text"],
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})


print(result)