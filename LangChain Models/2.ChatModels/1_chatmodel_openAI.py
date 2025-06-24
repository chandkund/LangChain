from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


load_dotenv()

chatmodel= ChatOpenAi(model = "gpt-4",temperature = 0 )
result = chatmodel.invoke("What is the capital of india?")
print(result.content)

