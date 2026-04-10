from langchain_community.tools import tool, StructuredTool
from pydantic import BaseModel, Field



class MultiplyInput(BaseModel):
    a: int = Field(description='The First number to multiply')
    b: int = Field(description='The Second number to multiply')



def get_multiplication(a: int, b: int) -> int:
    """ Return multiplication of two integers """
    return a * b



multiply_tool = StructuredTool.from_function(
    func=get_multiplication,
    name='Multiply',
    description='Multiply two integers',
    args_schema=MultiplyInput,
)



result = multiply_tool.invoke({'a': 2, 'b': 3})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)

print(multiply_tool.args_schema.model_json_schema())


