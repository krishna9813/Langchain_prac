import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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