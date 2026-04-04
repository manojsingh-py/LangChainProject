# LangChainProject

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

## 🛠 Tech Stack

| Technology   | Purpose                         |
| ------------ | ------------------------------- |
| Python       | Core programming language       |
| LangChain    | LLM orchestration               |
| Hugging Face | Open-source model inference     |
| Streamlit    | Web interface                   |
| dotenv       | Environment variable management |

---

## 📂 Project Structure

```
LangChainProject/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── langchain_env/
```

---

## ⚙️ Installation Guide

### Step 1: Clone Repository

```
git clone https://github.com/your-username/LangChainProject.git
cd LangChainProject
```

### Step 2: Create Virtual Environment

Windows:

```
python -m venv langchain_env
langchain_env\Scripts\activate
```

Mac/Linux:

```
python3 -m venv langchain_env
source langchain_env/bin/activate
```

---

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

---

## 🔐 Environment Variables Setup

Create a `.env` file in the root directory:

```
HUGGINGFACE_REPO_ID=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_TASK=text-generation
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```

Get your API key here:

https://huggingface.co/settings/tokens

---

## ▶️ Run the Application

Start the Streamlit server:

```
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧠 Model Used

Recommended model:

```
Qwen/Qwen2.5-7B-Instruct
```

Why this model:

* Strong reasoning capability
* Excellent instruction following
* Fast inference
* Free-tier compatible
* Ideal for structured research explanations

---

## 📸 Example Workflow

User selects:

Topic → RAG
Style → Technical
Length → Medium

App generates:

Structured explanation including:

* overview
* architecture
* workflow
* mathematical intuition
* practical examples

---

## 🔒 Security Best Practices

Never commit:

```
.env
langchain_env/
```

Ensure `.gitignore` contains:

```
.env
langchain_env/
__pycache__/
.idea/
.streamlit/
```

---

## 📌 Future Improvements

Planned enhancements:

* Upload research papers (PDF support)
* RAG-based document summarization
* Chat-style interaction interface
* Multi-model switching
* Export explanation to PDF
* Voice explanation support
* Deployment on Streamlit Cloud

---

## 🎯 Learning Outcomes from This Project

This project demonstrates:

* LangChain prompt engineering
* Hugging Face LLM integration
* Streamlit app development
* Environment variable security handling
* Modular AI application design

---

## 🤝 Contributing

Contributions are welcome.

Steps:

```
fork repository
create feature branch
commit changes
open pull request
```

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Manoj Singh

Built as part of a hands-on LangChain learning project exploring modern AI research assistants.
