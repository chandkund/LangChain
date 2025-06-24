
import asyncio
import sys

# Fix for Windows event loop bug in Streamlit + PyTorch
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

from langchain_core.prompts import PromptTemplate


# Load the Hugging Face pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False,
        do_sample=True  # important to match temperature
    )
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("🧠 Research Tool")



paper_input = st.selectbox("Select Research paper name",["Attention is all you need",
"Bert: Pre training of Deep Bidirectional Transfromers",
"GPT-3 Langauge Models are Few-shot learners",
"Diffusion models Beat GANs on Image Systhesis"])


length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragr)",
"Medium (3-5 Paragraphs)",
"long (detailed explation )"])


style_input = st.selectbox("Select Explanation Style",["Beginner- Friendly","Technical",
"Code- Oriented","Mathematical"])

#template 
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)



## Fill the placeholder 

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

