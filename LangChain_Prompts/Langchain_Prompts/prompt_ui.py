"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False
    )
)

model = ChatHuggingFace(llm=llm)


st.header("Reasearch Tool")

user_input = st.text_input("Enter your prompt")

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"})  # For CPU only

# Create HF pipeline
hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.5,
    max_new_tokens=150,
    return_full_text=False
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)
chat_model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("Research Tool")
user_input = st.text_input("Enter your prompt")

if st.button('Summarize'):
    formatted_prompt = f"### Instruction:\n{user_input}\n### Response:"
    result = chat_model.invoke(formatted_prompt)
    st.write(result.content)
