from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI


class GradeAnswer(BaseModel):
    """Binary score for generation answer."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question 
"""

# messages = [
#     SystemMessagePromptTemplate.from_template(system),
#     HumanMessagePromptTemplate.from_template("User question: \n\n {question} \n\n LLM generation: {generation}")
# ]
# answer_prompt = ChatPromptTemplate.from_messages(messages)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader


if __name__ == "__main__":
    print("testando módulo")
    print(answer_grader.invoke({"question": "Quanto é 2 + 2", "generation": "pizza"}))
