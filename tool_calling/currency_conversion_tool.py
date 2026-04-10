import requests
import os

from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Returns the conversion factor between base_currency and target_currency"""

    api_key = os.getenv("EXCHANGE_API_KEY")

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()["conversion_rate"]


@tool
def convert_currency(
        base_currency_value: int,
        conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Converts currency using conversion_rate"""

    return conversion_rate * base_currency_value



model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

llm_with_tool = model.bind_tools([
    get_conversion_factor,
    convert_currency
])


message = [
    HumanMessage(
        "What is the conversion factor between USD and INR? "
        "And convert 10 USD to INR."
    )
]


# ai_message = llm_with_tool.invoke(message)
#
# message.append(ai_message)
#
#
# conversion_rate = None
#
#
# for tool_call in ai_message.tool_calls:
#
#     if tool_call["name"] == "get_conversion_factor":
#
#         tool_message1 = get_conversion_factor.invoke(tool_call)
#
#         message.append(tool_message1)
#
#         conversion_rate = float(tool_message1.content)
#
#
#     if tool_call["name"] == "convert_currency":
#
#         tool_call["args"]["conversion_rate"] = conversion_rate
#
#         tool_message2 = convert_currency.invoke(tool_call)
#
#         message.append(tool_message2)

conversion_rate = None

while True:

    ai_message = llm_with_tool.invoke(message)
    message.append(ai_message)

    if not ai_message.tool_calls:
        break

    for tool_call in ai_message.tool_calls:

        if tool_call["name"] == "get_conversion_factor":

            tool_msg = get_conversion_factor.invoke(tool_call)

            conversion_rate = float(tool_msg.content)

            message.append(tool_msg)


        elif tool_call["name"] == "convert_currency":

            tool_call["args"]["conversion_rate"] = conversion_rate

            tool_msg = convert_currency.invoke(tool_call)

            message.append(tool_msg)


result = llm_with_tool.invoke(message)

print('result', result)
print('message', message)