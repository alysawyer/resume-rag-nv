from langchain_community.document_loaders import TextLoader

loader = TextLoader("test-input.rtf")
output = loader.load()

print(output)