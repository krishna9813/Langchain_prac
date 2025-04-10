import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "faiss_index")

if not os.path.exists(persistent_directory):
    print("DB does not exist, initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print('\n----- Document chunk information -----')
    print(f'Total number of chunks: {len(docs)}')
    print(f'Sample chunk:\n{docs[0].page_content}\n')

    print("\n-- Creating embeddings with HuggingFace --")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("-- Embeddings created --")

    print("\n-- Creating FAISS vector store --")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persistent_directory)
    print("-- Vector store created and saved locally --")

else:
    print("Vector store already exists. Loading from disk...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(persistent_directory, embeddings)
    print("-- Vector store loaded --")
