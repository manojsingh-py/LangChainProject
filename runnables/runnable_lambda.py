import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableSequence, RunnableParallel,
                                      RunnablePassthrough, RunnableLambda)


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
    template="Write a hindi funny joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

# Chain
joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({'topic': 'Santa-Banta'})

print(result)
print(f"Joke: {result['joke']}")
print(f"Word Count: {result['word_count']}")

# Chain Visualisation
chain.get_graph().print_ascii()

