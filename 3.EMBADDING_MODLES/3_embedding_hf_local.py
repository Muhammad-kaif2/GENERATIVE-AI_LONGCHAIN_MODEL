from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Islamabad is the capital of Pakistan",
"Lahore is the capital of Punjab",
"Karachi is the capital of Sindh",
"Peshawar is the capital of Khyber Pakhtunkhwa",
"Quetta is the capital of Balochistan"
]

vector = embedding.embed_documents(documents)

print(str(vector))