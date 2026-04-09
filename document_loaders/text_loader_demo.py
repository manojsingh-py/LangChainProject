from langchain_community.document_loaders import TextLoader

loader = TextLoader(r'cricket.txt', encoding='utf-8')

docs = loader.load()

print(len(docs))
print(docs)
print(docs[0])

print(docs[0].page_content)
print(docs[0].metadata)

