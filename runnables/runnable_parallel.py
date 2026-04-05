import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel



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
    template="Generate a tweet about the topic - {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a linkedin about the topic - {topic}",
    input_variables=["topic"],
)

# Parser
parser = StrOutputParser()

# Chain
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser),
})

result = parallel_chain.invoke({'topic': 'RAG'})

print(result)
print(f"Tweet:  {result['tweet']}")
print(f"Linkedin:  {result['linkedin']}")
