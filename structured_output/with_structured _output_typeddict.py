import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated
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
    summary: Annotated[str, 'A brief description of the review']
    sentiment: Annotated[str, 'The sentiment of the review either positive, negative or neutral']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I am not completely satisfied with my Samsung mobile experience. While the display quality is good, I noticed performance slowdowns after some months of use, especially when multiple apps run in the background. Many pre-installed apps cannot be removed, which consumes storage and affects overall performance. Extra background apps can also drain battery faster.""")

print(result)
print(f"Summary: {result['summary']}")
print(f"Sentiment: {result['sentiment']}")
