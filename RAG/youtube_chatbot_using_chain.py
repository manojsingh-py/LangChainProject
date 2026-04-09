import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv



load_dotenv()


def format_text(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Step 1: Indexing (Documnt Ingestion)

video_id = "EWvNQjAaOHw"
transcript_text = ''

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"],
        preserve_formatting=True
    )


    transcript_text = " ".join(
        [chunk.text for chunk in transcript_list]
    )

    # print(transcript_text)


except TranscriptsDisabled:
    transcript_text = []


# Step 1b: Indexing (Text Splitting)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(transcript_text)

# Step 1c & 1d: Indexing(Embedding gen, and store vector)

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)


# Step 2: Retrieval

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Step 3: Augmentation

prompt = PromptTemplate(
    template="""
    you are a helpful assistance.
    Answer only from the provided transcript context.
    If transcript is insufficient just say you don't know.
    
    {context}
    Question: {question}
    """,
    input_variables=['context', 'question'],

)

# Model
llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)


# Parser
parser = StrOutputParser()


#Chain
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_text),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | model | parser


# Final
result = main_chain.invoke('who is Andre')

print(result)

