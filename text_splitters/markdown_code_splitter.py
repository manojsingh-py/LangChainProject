from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

text = """# LangChainProject

## 📘 AI Research Tool using LangChain + Hugging Face + Streamlit

An interactive **AI-powered research assistant** that explains modern Artificial Intelligence topics in structured, customizable formats using open-source Large Language Models (LLMs).

This application allows users to select a research topic, choose explanation style and length, and receive a clear AI-generated explanation instantly.

---

## 🚀 Features

* 📚 Explains trending AI research topics
* 🎯 Multiple explanation styles

  * Beginner-friendly
  * Technical
  * Code-oriented
  * Mathematical
* 📏 Adjustable explanation length

  * Small
  * Medium
  * Long
* 🧠 Powered by open-source LLMs (Hugging Face)
* ⚡ Streamlit-based interactive UI
* 🔐 Secure API key handling via `.env`
* 🧩 Modular LangChain architecture

---

## 🧪 Supported Research Topics

The tool currently supports explanations for:

* Multimodal AI
* Agentic AI & Automation
* Generative AI & LLMs
* Computer Vision
* Reinforcement Learning
* Retrieval-Augmented Generation (RAG)

---
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(len(result))
print(result)


