import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

persist_dir = "chroma_db"
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    print("Loaded existing vectorstore from disk.")
else:
    print("No existing vectorstore found. Creating a new one...")

    # Load documents from 'data' folder
    docs = []
    for file in os.listdir("data"):
        file_path = os.path.join("data", file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # Create and persist vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_dir
    )
    print("Created new vectorstore and saved embeddings.")
