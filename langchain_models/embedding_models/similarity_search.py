from filetype import document_match
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    'Virat Kohli: A modern-day run-machine with an incredible average across formats, known as one of the best white-ball batsmen in history.',
    'MS Dhoni: The only captain to win all three ICC white-ball tournaments (World Cup, T20 World Cup, Champions Trophy), known for his calm demeanor and finishing skills',
    'Kapil Dev: Led India to their first-ever World Cup victory in 1983 and was arguably India greatest fast-bowling all-rounder.',
    'Sunil Gavaskar: The first player to score Test runs, known for his technical mastery against fast bowling in the 1970s and 80s.',
    'Sachin Tendulkar: Widely regarded as the "God of Cricket," he holds the record for the most international runs 34557 and centuries 100.'

]

text = 'Tell me about the MS'

doc_embedding = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(text)

scores = cosine_similarity([query_embedding], doc_embedding)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=False)[-1]

print(document[index])
