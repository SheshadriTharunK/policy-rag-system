from langchain.embeddings import HuggingFaceEmbeddings
from pydantic_ai import Agent
import os
import time
from dotenv import load_dotenv
from pydantic_ai.models.groq import GroqModel
from app.prompts import system_prompt
from langchain_community.vectorstores import FAISS
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

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
vector_db_dir = os.path.expanduser(
    "~\\Desktop\\Advintek\\data\\semantic-search\\index\\faiss"
)

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

