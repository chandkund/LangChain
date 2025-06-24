from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

HUGGINGFACEHUB_API_TOKEN = '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'


llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    temperature = 0.7,
    task = 'text-generation',
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN

)

model = ChatHuggingFace(llm = llm)

chat_hist = [
    SystemMessage(content= "You are a helpful AI assistant")
]
while True:
    user_input =  input("You: ")
    chat_hist.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_hist)
    chat_hist.append(AIMessage(content = result.content))
    print("AI: ",result.content)

print(chat_hist)


