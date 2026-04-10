import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


# -------------------------
# Load Model
# -------------------------

llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_REPO_ID"),
    task=os.getenv("HUGGINGFACE_TASK"),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)


# -------------------------
# Define Tool
# -------------------------

search_tool = DuckDuckGoSearchRun()


# -------------------------
# Create Agent
# -------------------------

agent = create_agent(
    model=model,
    tools=[search_tool]
)


# -------------------------
# Run Agent
# -------------------------

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capitlal of India and what is the population?"}]}
)

print(response)