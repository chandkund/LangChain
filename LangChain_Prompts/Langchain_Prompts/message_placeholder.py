from langchain_core.prompts import ChatpromptTemplate, MessagesPlaceholder 

## Chat Template 

chat_template = ChatPromptTemplate([
    ('system','you are a helful customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human','{query}')
])



chat_histroy = []
## Load Chat histroy 
with open('chat_histroy.txt') as f:
    chat_histroy.extend(f.readline())

print(chat_histroy)


## Create prompt 

prompt = chat_template.invoke({'chat_history':chat_histroy,'query':"where is my refund"})
print(prompt)