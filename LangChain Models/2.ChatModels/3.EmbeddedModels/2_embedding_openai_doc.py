from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(
    model= "text-embedding-3-large",
    dimension = 32
)


doc = [
    "Delhi is the capital city of india",
    "Kolkata is the capital of west bangal"
    "Paris is the capital of france"
]


result = embedding.embed_documents(doc)

print(str(result))





