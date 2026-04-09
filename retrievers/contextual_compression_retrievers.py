import os

from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

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


vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={"k":3, "lambda_mult":1},
)

# llm = HuggingFaceEndpoint(
#     repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
#     task=os.getenv('HUGGINGFACE_TASK'),
#     temperature=0.5
# )

llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')


compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever (
    base_compressor=compressor,
    base_retriever=retriever
)

query = 'what is fastapi'

res = compression_retriever.invoke(query)

print(res)