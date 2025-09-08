from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
   "Babar Azam is a Pakistani cricketer known for his consistent batting and elegant stroke play.",
"Shaheen Afridi is a Pakistani fast bowler famous for his deadly yorkers and new-ball swing.",
"Shahid Afridi, also known as 'Boom Boom', is remembered for his explosive batting and match-winning performances.",
"Shoaib Akhtar, nicknamed the 'Rawalpindi Express', is regarded as the fastest bowler in cricket history.",
"Wasim Akram, known as the 'Sultan of Swing', is one of the greatest left-arm fast bowlers of all time."
]

query = 'tell me about Babar Azam'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)