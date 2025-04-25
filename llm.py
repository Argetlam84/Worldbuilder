from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    collection_name="worldbuilding",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)


retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k":6,"lambda_mult":0.5}
    ) 

docs = retriever.invoke("Bana bir krallık hikayesi anlat")
for doc in docs:
    print(doc.page_content)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9,
    max_tokens=1000
)

system_prompt = (
    "You are a highly specialized AI assistant designed *only* for the purpose of worldbuilding.\n"
    "Your tasks include creating fictional worlds, characters, cultures, histories, and environments.\n\n"
    "You are explicitly forbidden from answering anything unrelated to worldbuilding.\n"
    "If the user asks about programming, real-world topics, translations, recipes, general knowledge, or other unrelated subjects,\n"
    "**firmly refuse** and say: 'I'm sorry, I can only assist with fictional worldbuilding tasks.'\n\n"
    "Use the following context to help with worldbuilding:\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human","{input}")
])



question_answer_chain = create_stuff_documents_chain(llm,prompt)

rag_chain = create_retrieval_chain(retriever,question_answer_chain)

response = rag_chain.invoke({"input":input("Create your first world!:")})

print(response["answer"])