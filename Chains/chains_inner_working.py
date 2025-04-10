from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda,RunnableSequence
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
translation_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a translator and convert the provided text into {language}"),
        ("human","Translate the following text to {language}:{text}"),
    ]
)
format_prompt=RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model=RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output=RunnableLambda(lambda x: x.content)
count_words=RunnableLambda(lambda x: f'Word count:{len(x.split())}\n{x}')
prepare_for_translation=RunnableLambda(lambda output:{"text":output,"language":"french"})

chain=format_prompt|invoke_model|StrOutputParser()|prepare_for_translation|translation_template|invoke_model|StrOutputParser()
result=chain.invoke({"animal":"Tiger","fact_count":2})
print(result)
# chain=RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)
# responses=chain.invoke({"animal":"Tiger","fact_count":5})
# print(responses)