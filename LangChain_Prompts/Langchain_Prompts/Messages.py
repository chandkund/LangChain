from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace 
from langchain_huggingface import HuggingFaceEndpoint 


HUGGINGFACEHUB_API_TOKEN = 'hf_AXsbYJPYSEmlS'

llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    temperature = 0.78,
    huggingfacehub_api_key = HUGGINGFACEHUB_API_TOKEN

)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = 'You are a helpful assistant'),
    HumanMessage(content = 'Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
print(messages)


