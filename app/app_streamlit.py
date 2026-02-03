
import streamlit as st
from rag import answer_question
import asyncio  

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
