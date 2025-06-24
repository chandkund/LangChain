

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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
result = model.invoke("### Instruction: What is the capital of jharkhand ?\n### Response:")
print(result.content)
print(result)  # See the raw result object

