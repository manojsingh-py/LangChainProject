from typing import Type

from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field



class MultiplyInput(BaseModel):
    a: int = Field(description='The First number to multiply')
    b: int = Field(description='The Second number to multiply')


class MultiplyTool(BaseTool):
    name: str = 'Multiply',
    description: str = 'Multiply two numbers together',
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()


result = multiply_tool.invoke({'a': 2, 'b': 3})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
# print(multiply_tool.args)

print(multiply_tool.args_schema.model_json_schema())


