
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from app.prompts import system_prompt
from langchain_community.vectorstores import FAISS
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import streamlit as st
import asyncio
from pathlib import Path
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from pydantic_ai import Agent

load_dotenv(override=True)
time.sleep(1)


# defining an agent with groq model
model  = GroqModel(
         model_name = "llama-3.1-8b-instant",
            provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY")))
rag_agent = Agent(model,
                  system_prompt=system_prompt)

# function to build context from vector DB
def build_context(vector_db: FAISS, query: str, top_k: int):
    base_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    hits = base_retriever.invoke(query)
    res = "\n".join([hit.page_content for hit in hits])
    return query + "\n<context>\n" + res + "\n</context>"

# loading the vector DB
vector_db_dir = "data/semantic-search/index/faiss"

# initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"token": os.getenv("HUGGING_FACE_TOKEN")},
)
# load vector DB from local directory
vector_db = FAISS.load_local(
    folder_path = vector_db_dir,
    embeddings = embeddings_model,
    allow_dangerous_deserialization = True,
)

# function to answer question using RAG approach
async def answer_question(question: str) -> str:
    query_with_context = build_context(
        vector_db = vector_db,
        query = question,
        top_k = 8,
    )
    print("Query with context:", query_with_context)
    agent = rag_agent
    result = await agent.run(query_with_context)
    return result.output 

# top of app_streamlit.py
import streamlit as st
import asyncio
from pathlib import Path
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from pydantic_ai import Agent

st.set_page_config(page_title="Policy System Advintek", layout="centered")
st.title("Policy System Advintek")


#  Sample questions
sample_questions = [
    "What is the leave policy?",
    "Tell me about the holiday policy.",
    "What is the code of conduct?",
    "How do I apply for work from home?",
"What should I do if I notice a conflict of interest in my team?",
"How does the company handle harassment or discriminatory behavior?",
"If a public holiday falls on a weekend, will it be carried forward or compensated?",
"How many flexible holidays can I take per year, and how do I choose them?",
"How many sick and casual leave days am I entitled to, and what’s the process to apply?",
"Can I encash my unused earned leave, and what is the maximum leave I can carry forward?",
"Which expenses are reimbursable for business travel, and what is the submission timeline?",
"Am I eligible for work from home, and what approvals are required?",
"Can I get extra casual leave if I’m assigned to a critical business project?",
"Under what circumstances can confidential company information be disclosed without prior authorization?"
]

# Streamlit selectbox for sample questions
selected_question = st.selectbox(
    "Choose a sample question or type your own:",
    ["--Type your own--"] + sample_questions
)

# Only show text input if user wants to type their own question
if selected_question == "--Type your own--":
    question = st.text_input("Enter your question:")
else:
    question = selected_question

# Ask button
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter or select a question!")
    else:
        # Run the async function in a synchronous Streamlit app
        answer = asyncio.run(answer_question(question))
        st.success("Answer:")
        st.write(answer)
