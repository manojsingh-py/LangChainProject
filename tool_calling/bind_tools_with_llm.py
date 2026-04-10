import os

from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv


load_dotenv()


@tool
def get_multiplication(a: int, b: int) -> int:
    """ Return multiplication of two integers """
    return a * b



result = get_multiplication.invoke({'a': 2, 'b': 3})

print(result)

print(get_multiplication.name)
print(get_multiplication.description)
print(get_multiplication.args)

print(get_multiplication.args_schema.model_json_schema())


llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)


llm_with_tool = model.bind_tools([get_multiplication])


print(f'llm_with_tool: {llm_with_tool}')
