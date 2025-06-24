from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

embedding = OpenAIEmbeddings(
    model= 'text-embedding-3-large',
    dimension = 300   
)

document = [
    "Virat kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "Ms Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachine Tendulkar, also known as the 'God of criclet' holds many battting records."
    "Rohit Sharma is known for his elegent batting and record-breaking double centuries."
    "Jasprit Bumrah is an indian fast bowler knowm for his unothodox action and yorkers."
]


query = "Tell me about virat kohli"



doc_embeddings = embedding.embed_documents(document)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]
index, score = sorteed(list(enumerate(scores)),key= lambda x:x[1])[-1]

print(query)
print(document[index])
print("Similarity score is:",score)
