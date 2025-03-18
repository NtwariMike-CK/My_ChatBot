import os
import shutil
from typing import List

# Load packages for getting data from pdf, text etc
from langchain_community.document_loaders import DirectoryLoader

# Load packages for splitting data into chunks
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load packages for creating vectors and storing data into chromadb
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from langchain_community.vectorstores import Chroma

# Initialize directory paths
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()  # In Colab, use the current working directory

DATA_PATH = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Ensure the paths exist
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Directory NOT found: {DATA_PATH}")
print(f"✅ Documents directory found: {DATA_PATH}")

# Initialize embedding model function
def get_embeddings():
    """Initialize and return the embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Important: ensure embeddings are normalized
    )
    return embeddings

# Load documents from the specified directory
def load_documents():
    """Load markdown documents from the data directory"""
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

# Split documents into smaller chunks for better retrieval
def split_text(documents: List[Document]):
    """Split documents into smaller chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Print debug information about chunk sizes
    if chunks:
        print(f"Sample chunk size: {len(chunks[0].page_content)} characters")
        print(f"Sample chunk content: {chunks[0].page_content[:100]}...")
    
    return chunks

# Store document chunks in ChromaDB
def store_in_chromadb(chunks: List[Document]):
    """Store document chunks in ChromaDB for vector search"""
    # Initialize embeddings with normalization
    embeddings = get_embeddings()
    
    # Remove existing database if it exists to avoid conflicts
    if os.path.exists(CHROMA_PATH):
        print(f"🔄 Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    # Create directory for the DB
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return vector_store

# Retrieve documents from ChromaDB for testing
def retrieve_from_chromadb(query, k=3):
    """Test retrieval from ChromaDB"""
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)

    print(f"🔹 Retrieved {len(retrieved_docs)} Documents for query: '{query}'")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...")

    return retrieved_docs

# Main function to run the entire pipeline
def main():
    """Run the document ingestion pipeline"""
    print("📚 Starting document ingestion pipeline...")
    documents = load_documents()
    chunks = split_text(documents)
    store_in_chromadb(chunks)
    print("✅ Ingestion complete. Testing retrieval...")
    
    # Test retrieval with a few sample queries
    test_queries = ["random password", "projects", "education"]
    for query in test_queries:
        print(f"\n🔍 Testing retrieval for: '{query}'")
        retrieve_from_chromadb(query, k=2)

if __name__ == "__main__":
    main()
