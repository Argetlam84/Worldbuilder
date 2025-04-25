import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

with open("datasets/dataset.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

splitter = RecursiveJsonSplitter(max_chunk_size=10000)
docs = splitter.create_documents(texts=[data], convert_lists=True)
print(len(docs))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    collection_name="worldbuilding",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

batch_size=100

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    try:
        vector_store.add_documents(documents=batch)
        print(f"Batch {i // batch_size + 1} added")
        time.sleep(45)
    except Exception as e:
        print(f"Batch {i // batch_size + 1} error: {e}")
