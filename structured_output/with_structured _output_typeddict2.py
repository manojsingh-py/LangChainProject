import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

# Schema
class Review(TypedDict):
    key: Annotated[list[str], 'Write all the keys discussed in this reviews']
    summary: Annotated[str, 'A brief description of the review']
    sentiment: Annotated[str, 'The sentiment of the review either positive, negative or neutral']
    pros: Annotated[Optional[list[str]], 'Write all the pros in this review']
    cons: Annotated[Optional[list[str]], 'Write all the cons in this review']
    name: Annotated[str, 'The name of the reviewer']

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


