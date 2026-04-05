from typing import TypedDict

class AIMessage(TypedDict):
    text: str
    number: int
    date: str


ai_message = AIMessage(text="Hello World", number=5, date="2020-07-21")

print(ai_message)