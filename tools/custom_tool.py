from langchain_community.tools import tool



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


