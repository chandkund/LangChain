from langchain_core.prompts import ChatpromptTemplate 
from langchain_core.message import HumanMessage,SystemMessage 


chat_template = ChatPromptTemplate([
    ('system','You are a helpful{domain} expert'),
    ('human','Expalin in simple temrs, what is {topic}')
])

prompt  = chat_template.invoke({
    'domain':'cricket',
    'topic':'Dusra'
    })

print(prompt)
