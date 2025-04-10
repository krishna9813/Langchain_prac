from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel
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
summary_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie critic"),
        ("human","Provide a brief summary of the movie {movie_title} and language of the film is {language}."),
    ]
)
def analyse_plot(plot):
    plot_template=ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic"),
            ("human","Analyze the plot:{plot}.What are the pros and cons??"),
        ]
    )
    return plot_template.format_prompt(plot=plot)
def analyze_character(character):
    character_template=ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic"),
            ("human","Analyze the characters :{character}. What are the strength and weaknesses??"),
        ]
    )
    return character_template.format_prompt(character=character)
def combine_verdicts(plot_analysis,character_analysis):
    return f"Plot analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

plot_branch_chain=(
    RunnableLambda(lambda x:analyse_plot(x))|llm|StrOutputParser()
)
character_branch_chain=(
    RunnableLambda(lambda x : analyze_character(x))|llm|StrOutputParser()
)

chain=(
    summary_template|
    llm|
    StrOutputParser()|
    RunnableParallel(branches={"plot":plot_branch_chain,"characters":character_branch_chain})|
    RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"],x["branches"]["characters"]))
)
result=chain.invoke({"movie_title":"Master","language":"Tamil"})
print(result)