import os

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

# Prompts
prompt1 = PromptTemplate(
    template="Generate detailed report on given {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate 5 bullet points from the following text:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()


# Chain
chain = (
    prompt1
    | model
    | parser
    #| (lambda x: {"text": x})
    | prompt2
    | model
    | parser
)


result = chain.invoke({"topic": "Cricket"})

print(result)


# Chain Visualisation
chain.get_graph().print_ascii()