import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough



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
joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain_joke': RunnableSequence(prompt2, model, parser),
})

chain = RunnableSequence(joke_generator_chain, parallel_chain)

res = chain.invoke({'topic': 'Santa-Banta'})

print(res)