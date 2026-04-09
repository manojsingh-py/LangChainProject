from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()

print(docs)
print(len(docs))
print(docs[0])