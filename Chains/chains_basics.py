from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
import os
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("Error")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="meta-llama/llama-3.3-70b-instruct:free"
    )

prompt_template=ChatPromptTemplate.from_messages(
    [
       ("system","You are a facts expert who knows fact about {animal}."),
       ("human","Tell me {fact_count} facts"),
    ]
 )
chain=prompt_template|llm|StrOutputParser()
result=chain.invoke({"animal":"elephant","fact_count":5})
print(result)