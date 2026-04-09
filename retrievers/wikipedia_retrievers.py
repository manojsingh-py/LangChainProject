from langchain_community.retrievers.wikipedia import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = 'who is the lal bahadur shastri'

docs = retriever.invoke(query)

print(docs)