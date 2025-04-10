from langchain_openai import ChatOpenAI  
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType,initialize_agent,load_tools
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
def generate_pet_names(animal_type,pet_color):


    try:
      prompt_template =PromptTemplate(
            input_variables=['animal_type','pet_color'],
            template=f"I have a {animal_type} pet with color {pet_color} and I want a cool name for it.Suggest me five cool names for my pet."
         )
      name_chain=LLMChain(llm=llm,prompt=prompt_template,output_key="  ")
      response=name_chain({'animal_type':animal_type,'pet_color':pet_color})
      return response

    except Exception as e:
        return f"Error: {e}"
def langchain_agent():
    tools=load_tools(["wikipedia","llm-math"],llm=llm)
    agent=initialize_agent(
        tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True
    )
    result=agent.run(
        "What is the average age of a cat? Multiply the age by 15"
    )
    print(result)
def storeChat():
    chat_history=[]
    system_message=SystemMessage(content="You are a helpful AI assistant")
    chat_history.append(system_message)
    while True:
        query=input("you: ")
        if query.lower()=="exit":
            break
        chat_history.append(HumanMessage(content=query))

        answer=llm.invoke(chat_history)
        res=answer.content
        chat_history.append(AIMessage(content=res))
        
        print(f'AI :{res}')
    print("------MESSAGE HISTORY------")
    print(chat_history)

if __name__=="__main__":
    storeChat()
