import os
from functools import partial

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Schema
class Person(BaseModel):

    name: str = Field(description="Name of person")
    age: int  = Field(description="Age of person")
    city: str = Field(description="City of person")
    country: str = Field(description="Country of person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='generate the a name, age, city, country of a fictional {place} person \n '
             'format_instructional',
    input_variables=['place'],
    partial_variables={'format_instructional': parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'India'})

res = model.invoke(prompt)

final_res = parser.parse(res.content)

print(final_res)


# using chain

chain = template | model | parser

res = chain.invoke({'place': 'Japan'})

print(res.content)