from langchain_huggingface import HuggingFaceEmbeddings
from pydantic_ai import Agent
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai.models.groq import GroqModel
from langchain_community.vectorstores import FAISS
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from fastapi import FastAPI
from app.rag import answer_question

load_dotenv(override=True)
time.sleep(1)
system_prompt = """
You are a company policy assistant.

Your task is to answer employee questions STRICTLY using the provided policy context.
Do NOT use outside knowledge.
Do NOT make assumptions.

If the answer is not explicitly present in the context, respond exactly with:
"I could not find this information in the company policies."

--------------------
Examples:

Context:
"Employees are entitled to 10 days of sick leave per year."

Question:
"How many sick leave days are employees allowed?"

Answer:
"Employees are entitled to 10 days of sick leave per year."

---

Context:
"Employees must obtain prior approval from their reporting manager before opting for WFH."

Question:
"Do employees need approval to work from home?"

Answer:
"Yes, employees must obtain prior approval from their reporting manager before opting for work from home."

---

Context:
"The following holidays are observed across all company locations:
Republic Day – January 26
Independence Day – August 15"

Question:
"Is Republic Day a company holiday?"

Answer:
"Yes, Republic Day on January 26 is observed as a company holiday across all company locations."
---

Context:
"There is no information in the provided policies about employee stock options."

Question:
"Do employees receive stock options?"

Answer:
"I could not find this information in the company policies."

--------------------
Only answer using the given context.
"""


# defining an agent with groq model
model  = GroqModel(
         model_name = "llama-3.1-8b-instant",
            provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY")))
rag_agent = Agent(model,
                  system_prompt=system_prompt)

# function to build context from vector DB
def build_context(vector_db: FAISS, query: str, top_k: int):
    print("Building context for query:", vector_db)
    base_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    hits = base_retriever.invoke(query)
    print("Hits retrieved:", hits)
    res = "\n".join([hit.page_content for hit in hits])
    return f"""
        Context:
        {res}

        Question:
        {query}
        """

# loading the vector DB
BASE_DIR = Path(__file__).resolve().parent.parent

vector_db_dir = BASE_DIR / "data" / "semantic-search"/"index"/"faiss"
print(f"Loading vector DB from {vector_db_dir}")
# initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"token": os.getenv("HUGGING_FACE_TOKEN")},
)
# load vector DB from local directory
import os
print(os.getcwd())
vector_db = FAISS.load_local(
    folder_path = vector_db_dir,
    embeddings = embeddings_model,
    allow_dangerous_deserialization = True,
)
docs = vector_db.similarity_search("holiday", k=5)
for i, d in enumerate(docs):
    print(f"\n--- Doc {i} ---")
    print(d.page_content)


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

application = FastAPI(title="Policy System Advintek")

# class QuestionRequest(BaseModel):
#     question: str

@application.post("/ask")
async def ask_question(question: str):
    result = await answer_question(question)  
    lines = result.replace("\n", "")
    return {"answer": lines}

