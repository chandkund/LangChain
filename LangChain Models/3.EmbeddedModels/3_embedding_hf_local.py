from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)


text = "Delhi is the the capital city of india"

vector = embedding.embed_query(text)
print(str(vector))
print("________________________________________________________")


doc = [
    "Delhi is the capital city of india",
    "Kolkata is the capital of west bangal"
    "Paris is the capital of france"
]


vector = embedding.embed_documents(doc)
print(str(vector))