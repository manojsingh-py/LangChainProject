import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableSequence, RunnableLambda,
                                      RunnableBranch, RunnablePassthrough)

def word_count(text):
    return len(text.split())

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
    template="Write a detail report on this topic - {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Summerize the following report - {text}",
    input_variables=["text"],
)

# Parser
parser = StrOutputParser()

# Chain
report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen_chain, branch_chain)

result = chain.invoke({'topic': 'RAG'})

print(result)