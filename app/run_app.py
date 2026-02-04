from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question

application = FastAPI(title="Policy System Advintek")

# class QuestionRequest(BaseModel):
#     question: str

@application.post("/ask")
async def ask_question(question: str):
    result = await answer_question(question)  
    lines = result.replace("\n", "")
    return {"answer": lines}