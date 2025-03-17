import os
# Load packages for getting data from pdf, text etc
from langchain_community.document_loaders import DirectoryLoader

# Load packages for spliting data into chunks
from langchain.schema import Document  # Import the correct type
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load packages for creating vectors and storing data into chromadb
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from langchain_community.vectorstores import Chroma


# Initialise the dir with data
# DATA_PATH = os.path.abspath("Final chatbot/rag project/docs/")

# Initialise the dir with data
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()  # In Colab, use the current working directory

DATA_PATH = os.path.join(BASE_DIR, "docs")

# Ensure the path exists before proceeding
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Directory NOT found: {DATA_PATH}")

print(f"✅ Directory found: {DATA_PATH}")


# Initialize ChromaDB client (Persistent storage) the location to store the db
# Define the path to the database folder dynamically

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Ensure the path exists or create it
os.makedirs(CHROMA_PATH, exist_ok=True)

print(f"✅ Database path set to: {CHROMA_PATH}")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def retrieve_from_chromadb(query, k=3):
  """Retrieves relevant documents from ChromaDB based on the query."""
  vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
  )
  retriever = vector_store.as_retriever(search_kwargs={"k": k})
  retrieved_docs = retriever.get_relevant_documents(query)

  print("🔹 Retrieved Documents:")
  for doc in retrieved_docs:
    print(doc.page_content)

  return retrieved_docs


if __name__ == "__main__":
  while True:
    query_text= input("Search Keyword")
    if query_text.lower() == "exit" or query_text == 1:
      break
    else:
      print(retrieve_from_chromadb(query_text))

