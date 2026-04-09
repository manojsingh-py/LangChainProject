import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# Step 1: Indexing (Document Ingestion)

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

# print(len(chunks))
# print(chunks[0])


# Step 1c & 1d: Indexing(Embedding gen, and store vector)

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids('2781bee2-3321-47ae-9763-c145b804901d'))


# Step 2: Retrieval

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kargs={"k": 4}
)

# res = retriever.invoke('what is deep seek')

# print(res)

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

llm = HuggingFaceEndpoint(
    repo_id=os.getenv('HUGGINGFACE_REPO_ID'),
    task=os.getenv('HUGGINGFACE_TASK'),
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

question = 'is the topic about deep seek discussed in this video? if yes what was discussed?'

retrieved_docs = retriever.invoke(question)

context = '\n'.join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':context, 'question':question})


result = model.invoke(final_prompt)

print(result.content)
