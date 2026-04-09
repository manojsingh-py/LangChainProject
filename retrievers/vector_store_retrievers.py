from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

documents = [
    Document(
        page_content="FastAPI is a modern web framework for building APIs with Python.",
        metadata={"source": "fastapi_docs", "topic": "web_framework"}
    ),

    Document(
        page_content="LangChain helps developers build applications powered by large language models.",
        metadata={"source": "langchain_docs", "topic": "llm_framework"}
    ),

    Document(
        page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.",
        metadata={"source": "faiss_docs", "topic": "vector_database"}
    ),

    Document(
        page_content="Dependency Injection in FastAPI allows reusable components like database sessions and authentication logic.",
        metadata={"source": "fastapi_dependency", "topic": "architecture"}
    ),

    Document(
        page_content="Semantic chunking splits text based on meaning instead of character length.",
        metadata={"source": "langchain_splitters", "topic": "text_processing"}
    ),
]


embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='vs_retriever',
)

retriever = vector_store.as_retriever(search_kwargs={'k':1})

query = 'what is fastapi'

res = retriever.invoke(query)

print(res)