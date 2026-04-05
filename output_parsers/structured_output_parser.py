import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Schema
schema = [
    ResponseSchema(name='fact1', description='fact1 about the topic'),
    ResponseSchema(name='fact2', description='fact2 about the topic'),
    ResponseSchema(name='fact3', description='fact3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt template
template = PromptTemplate(
    template="""
Write 3 facts about {topic}

{format_instructions}
""",
    input_variables=['topic'],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    },
)

prompt = template.invoke({'topic': 'Black hole'})

# Model response
result = model.invoke(prompt)

# Parse structured output
final_result = parser.parse(result.content)

print(final_result)


# Using chain

chain = template | model | parser

res = chain.invoke({'topic': 'Black hole'})

print(res.content)
