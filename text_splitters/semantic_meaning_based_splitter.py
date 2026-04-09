from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()


text = """
Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms.Cricket is one of the world's most popular outdoor sports, celebrated for its rich history and strategic depth. Often called the "Gentleman's Game," it originated in 16th-century England and has since become a cultural phenomenon, especially in countries like India, Australia, and Pakistan.

Terrorism is the calculated use of violence, fear, and intimidation against civilians to achieve political, ideological, or religious aims. It is a global menace that violates fundamental human rights, disrupts peace, and causes widespread destruction. Terrorist activities range from local attacks to international networks, often targeting innocent people to cause maximum chaos and destabilize governments. Farming is the essential practice of cultivating land, raising crops, and rearing animals to provide food and raw materials. Farmers work tirelessly to produce staples like rice, wheat, and vegetables, often waking up before dawn. It is a demanding profession that feeds the nation and supports the economy, relying on both traditional knowledge and modern technology. 

Air pollution is the contamination of the atmosphere by harmful gases, dust, and smoke, primarily from factories, vehicles, and burning waste. It lowers air quality, causing severe health issues like asthma and heart disease. This environmental crisis also drives climate change and damages ecosystems. 
"""

splitter = SemanticChunker(
    HuggingFaceEmbeddings(),
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=-1
)


docs = splitter.create_documents([text])

print(len(docs))
print(docs)


