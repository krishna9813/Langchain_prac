import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


cuurent_dir=os.path.dirname(os.path.abspath(__file__))
persistent_directory=os.path.join(cuurent_dir,"db","faiss_index")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(persistent_directory, embeddings, allow_dangerous_deserialization=True)

query="Where does Gandalf meet Frodo?"

retriever=db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={"k":3,"score_threshold":0.5},
)
relevant_docs=retriever.invoke(query)

print("\n---Relevant Documents---")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

combined_input=(
    "Here are some documents that might help answer the question: "
    +query
    +"\n\nRelevant Document:\n"
    +"\n\n".join([doc.page_content for doc in relevant_docs])
    +"\n\n Please provide a rough answer based only on the provided information.If the answer is not found in the documents , respond with Im not sure."
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("Error")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="meta-llama/llama-3.3-70b-instruct:free"
    )
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]
result=llm.invoke(messages)
print(result.content)