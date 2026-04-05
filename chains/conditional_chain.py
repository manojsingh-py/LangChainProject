import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description='sentiment of the feedback text')

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment  of the following feedback text into positive or negative \n {feedback_text} \n {format_instruction}',
    input_variables=['feedback_text'],
    partial_variables={'format_instruction': parser2.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template='Write an appropriate response of the positive feedback \n {feedback_text}',
    input_variables=['feedback_text'],
)

prompt3 = PromptTemplate(
    template='Write an appropriate response of the negative feedback \n {feedback_text}',
    input_variables=['feedback_text'],
)
# classifier
classifier_chain = prompt1 | model | parser2

# branch logic
branch_chain = RunnableBranch(
    (lambda x:x.sentiment  == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment  == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not found sentiment ')
)

chain = classifier_chain | branch_chain

# result = chain.invoke({'feedback_text': "This is wonderful phone"})
result = chain.invoke({'feedback_text': "This is worse phone"})

print(result)

# Visualize chain

chain.get_graph().print_ascii()