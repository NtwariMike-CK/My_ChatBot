import os
import shutil
import uuid
from typing import List, Dict, Any

# Load packages for getting data from pdf, text etc
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader

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
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Better for Q&A tasks
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Ensure embeddings are normalized
    )
    return embeddings

# Load documents from the specified directory
def load_documents():
    """Load markdown and text documents from the data directory"""
    # Create a loader that can handle both markdown and text files
    loaders = {
        ".md": DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        ".txt": DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    }
    
    documents = []
    for ext, loader in loaders.items():
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"✅ Loaded {len(loaded_docs)} {ext} documents")
        except Exception as e:
            print(f"⚠️ Error loading {ext} documents: {e}")
    
    if not documents:
        print("⚠️ No documents were loaded. Check file types and permissions.")
    else:
        print(f"✅ Loaded {len(documents)} total documents from {DATA_PATH}")
    
    # Add document IDs for better tracking
    for i, doc in enumerate(documents):
        if 'source' in doc.metadata:
            doc.metadata['doc_id'] = f"{os.path.basename(doc.metadata['source'])}_{i}"
        else:
            doc.metadata['doc_id'] = f"doc_{i}_{uuid.uuid4().hex[:8]}"
    
    return documents

# Split documents into smaller chunks for better retrieval
def split_text(documents: List[Document]):
    """Split documents into smaller chunks with overlap"""
    # Create a resume-specific text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for resume-specific content
        chunk_overlap=150,  # Higher overlap for better context preservation
        length_function=len,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Ensure chunks maintain document IDs and add section metadata
    for i, chunk in enumerate(chunks):
        # Keep the original doc_id
        if 'doc_id' not in chunk.metadata and 'source' in chunk.metadata:
            chunk.metadata['doc_id'] = f"{os.path.basename(chunk.metadata['source'])}_{i}"
        
        # Add chunk ID
        chunk.metadata['chunk_id'] = f"chunk_{i}"
        
        # Try to identify the section based on content
        content = chunk.page_content.lower()
        if any(keyword in content for keyword in ["education", "university", "degree", "college"]):
            chunk.metadata['section'] = "education"
        elif any(keyword in content for keyword in ["experience", "work", "job", "position"]):
            chunk.metadata['section'] = "experience"
        elif any(keyword in content for keyword in ["skill", "proficiency", "technology", "framework"]):
            chunk.metadata['section'] = "skills"
        else:
            chunk.metadata['section'] = "general"
    
    # Print debug information
    if chunks:
        print(f"Sample chunk size: {len(chunks[0].page_content)} characters")
        print(f"Sample chunk content: {chunks[0].page_content[:100]}...")
        print(f"Sample chunk metadata: {chunks[0].metadata}")
    
    return chunks

# Store document chunks in ChromaDB
def store_in_chromadb(chunks: List[Document]):
    """Store document chunks in ChromaDB for vector search"""
    # Initialize embeddings
    embeddings = get_embeddings()
    
    # Remove existing database if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"🔄 Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    # Create directory for the DB
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Create and persist the vector store with richer metadata
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
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
    
    # Test with both similarity search and MMR
    print(f"\n🔍 Testing standard similarity search for: '{query}'")
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    print(f"🔹 Retrieved {len(retrieved_docs)} Documents")
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"\nDocument {i+1} (Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Section: {doc.metadata.get('section', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...")
    
    # Test with MMR for diversity
    print(f"\n🔍 Testing MMR search for: '{query}'")
    mmr_docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
    
    print(f"🔹 Retrieved {len(mmr_docs)} Documents with MMR")
    for i, doc in enumerate(mmr_docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Section: {doc.metadata.get('section', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...")
    
    return retrieved_docs, mmr_docs

# Main function to run the entire pipeline
def main():
    """Run the document ingestion pipeline"""
    print("📚 Starting document ingestion pipeline...")
    documents = load_documents()
    chunks = split_text(documents)
    store_in_chromadb(chunks)
    print("✅ Ingestion complete. Testing retrieval...")
    
    # Test retrieval with sample queries relevant to resumes
    test_queries = [
        "What education does the candidate have?",
        "What are the candidate's technical skills?",
        "What work experience does the candidate have?",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing retrieval for: '{query}'")
        retrieve_from_chromadb(query, k=2)

if __name__ == "__main__":
    main()
