from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import os

if os.path.exists(".env"):
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    GOOGLE_API_KEY = st.text_input("Enter your Google API KEY", type="password")
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Worldbuilding Assistant 🌍✨")

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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9,
    max_tokens=1500
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input=st.chat_input("Create your first world!")
prompt = user_input

system_prompt = (
    "You are a highly specialized AI assistant designed *only* for the purpose of worldbuilding.\n"
    "Your tasks include creating fictional worlds, characters, cultures, histories, environments, and lore.\n"
    "You are also capable of designing game worlds. Based on the universe of the game,\n"
    "you help with level design, quest structure, and world mechanics.\n"
    "You also write unique backstories for characters, fitting the world they belong to.\n\n"
    "You are skilled in the role of a Dungeon Master (DM) for tabletop games like Dungeons & Dragons.\n"
    "You can design campaigns, encounters, magical systems, and roleplay scenarios—all rooted in immersive worldbuilding.\n\n"
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

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Building your world..."):
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
