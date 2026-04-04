from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = 'What is the capital of India?'
doc = [
    'Delhi',
    'India',
    'Manoj'
]
vector = embedding.embed_query(text)

doc_vec = embedding.embed_documents(doc)

print(vector)
print(doc_vec)

