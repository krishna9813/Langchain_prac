from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="meta-llama/llama-3.3-70b-instruct:free"
)


positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}")
])
negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}")
])
neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}")
])
escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}")
])


feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a feedback generator for specified products."),
    ("human", "Generate a random feedback either positive, negative, or neutral on the product: {product}")
])
classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}")
])


feedback_chain = feedback_template | llm | StrOutputParser()


classification_chain = (
    RunnableLambda(lambda feedback: {"feedback": feedback}) |
    classification_template |
    llm |
    StrOutputParser()
)


branches = RunnableBranch(
    (lambda x: "positive" in x["sentiment"].lower(),
     positive_feedback_template | llm | StrOutputParser()),
    (lambda x: "negative" in x["sentiment"].lower(),
     negative_feedback_template | llm | StrOutputParser()),
    (lambda x: "neutral" in x["sentiment"].lower(),
     neutral_feedback_template | llm | StrOutputParser()),
    escalate_feedback_template | llm | StrOutputParser()  
)


full_chain = (
    RunnableLambda(lambda x: {"product": x["product"]}) |
    feedback_chain |
    RunnableLambda(lambda feedback: {"feedback": feedback, "sentiment": classification_chain.invoke({"feedback": feedback})}) |
    RunnableLambda(lambda x: {
        "feedback": x["feedback"],
        "sentiment": x["sentiment"],
        "response": branches.invoke(x)
    })
)


product_name = "OnePLUS 12R"
result = full_chain.invoke({"product": product_name})


print(" Generated Feedback:\n", result["feedback"])
print("\nSentiment Classification:\n", result["sentiment"])
print("\n Final Response:\n", result["response"])
