import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader



load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Doc. Loader
loader = TextLoader(r'cricket.txt', encoding='utf-8')

docs = loader.load()


# Prompts
prompt = PromptTemplate(
    template="Write a summary for following poem - {poem}",
    input_variables=["poem"],
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})


print(f"Poem: {docs[0].page_content}")

print(result)


