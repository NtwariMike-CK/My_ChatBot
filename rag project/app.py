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
DATA_PATH = os.path.abspath("Final chatbot/rag project/docs/")



# Initialize ChromaDB client (Persistent storage) the location to store the db
CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Function for running all
def main():
  documents = load_documents()
  chunks = split_text(documents)
  store_in_chromadb(chunks)





def load_documents():
  loader = DirectoryLoader(DATA_PATH, glob="*.md")
  documents = loader.load()
  return documents



def split_text(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
    length_function =len,
    add_start_index = True,
    separators=["\n\n", "\n", " ", ""]  # Prioritizes splitting by paragraphs, then newlines
  )
  chunks = text_splitter.split_documents(documents)  # Correct: Use already loaded documents

  print(f"split {len(documents)} documents into {len(chunks)} chunks")

  document = chunks[3]
  print(document.page_content)
  print(document.metadata)

  return chunks


def store_in_chromadb(chunks: list[Document]):
  """Stores document chunks into ChromaDB for retrieval."""
  vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory = CHROMA_PATH
  )
  vector_store.persist()
  print("✅ Successfully stored chunks in ChromaDB")
  return vector_store


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
  main()
  # while True:
  #   query_text= input("Search Keyword")
  #   if query_text.lower() == "exit" or query_text == 1:
  #     break
  #   else:
  #     print(retrieve_from_chromadb(query_text))

