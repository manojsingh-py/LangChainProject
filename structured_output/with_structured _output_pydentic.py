import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Schema
class Review(BaseModel):

    key_theme: list[str] = Field(description='Write all the keys discussed in this reviews')
    summary: str = Field(description='Write all the summary discussed in this reviews')
    sentiment: str = Field(description='Write all the sentiment discussed in this reviews')
    pros: Optional[list[str]] = Field(description='Write all the pros in this review')
    cons: Optional[list[str]] = Field(description='Write all the cons in this review')
    name: str = Field(description='Write all the name of this reviewer')


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Samsung smartphones have consistently remained among the most influential devices in the Android ecosystem, offering a wide spectrum of models across budget, mid-range, and premium categories. While they deliver strong innovation and reliability in several areas, they also present certain limitations that affect long-term user experience depending on the model category and usage patterns.
Pros: Samsung’s AMOLED and Dynamic AMOLED displays are widely considered among the best in the smartphone industry. They offer excellent contrast ratios, deep blacks, high brightness levels, and smooth refresh rates, making them ideal for media consumption, gaming, and productivity tasks.
Cons: Samsung devices often include duplicate apps alongside Google services, which can occupy storage space and occasionally affect system efficiency. Although some can be disabled, not all can be removed completely.
Reviewed by XYZ
""")

print(result)
print(f"Summary: {result['summary']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Pros: {result['pros']}")
print(f"Cons: {result['cons']}")
print(f"Name: {result['name']}")


