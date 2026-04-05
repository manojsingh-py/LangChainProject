import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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
    template="Generate short and simple notes on following text:\n{text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from following text:\n{text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="""
            Merge the provided notes and quiz into a single document.
            Notes:{notes}
            Quiz:{quiz}
            """,
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

# Parallel execution
parallel_chain = RunnableParallel(
    notes=prompt1 | model | parser,
    quiz=prompt2 | model | parser
)

# Merge execution
merged_chain = prompt3 | model | parser

# Final pipeline
chain = parallel_chain | merged_chain


text = """REST framework includes an abstraction for dealing with ViewSets..."""

result = chain.invoke({"text": text})

print(result)


# Visualize chain
chain.get_graph().print_ascii()