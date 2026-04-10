from langchain_core.tools import tool, BaseTool


@tool
def multiply_tool(a: int, b: int) -> int:
    """ Multiply two numbers """
    return a * b


@tool
def addition_tool(a: int, b: int) -> int:
    """ Add two numbers """
    return a + b




class MathToolkit:

    def get_tools(self):
        return [addition_tool, multiply_tool]


toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    result = tool.invoke({'a': 2, 'b': 3})

    print(result)

    print(multiply_tool.name)
    print(multiply_tool.description)


