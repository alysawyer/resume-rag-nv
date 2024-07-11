from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="ai-llama3-70b")

response = llm.invoke("Tell me a story about GPUs")
print(response)