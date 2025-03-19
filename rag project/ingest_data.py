import os
import shutil
import uuid
from typing import List

# packages for web scalping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Document loading packages
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredURLLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector embedding and storage
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

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)

# Define file paths - adjust these to your actual file locations
RESUME_PDF_PATH = os.path.join(DATA_PATH, "Ntwari_Mike_Resume.pdf")
RESUME_URL = "https://ntwarimike-ck.github.io/My-resume/"
ADDITIONAL_URLS = [
    # Add any additional pages from your website if needed
    "https://ntwarimike-ck.github.io/My-resume/#about",
    "https://ntwarimike-ck.github.io/My-resume/#experience",
    "https://ntwarimike-ck.github.io/My-resume/#projects",
]

def get_embeddings():
    """Initialize and return the embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight model good for resume data
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def load_pdf_document():
    """Load the resume PDF as a complete document"""
    if not os.path.exists(RESUME_PDF_PATH):
        print(f"⚠️ Resume PDF not found at {RESUME_PDF_PATH}")
        return []
    
    print(f"📄 Loading resume PDF from {RESUME_PDF_PATH}")
    loader = PyPDFLoader(RESUME_PDF_PATH)
    documents = loader.load()
    
    # Add metadata to identify this as resume content
    for doc in documents:
        doc.metadata['source_type'] = 'resume_pdf'
        doc.metadata['source'] = 'Ntwari_Mike_Resume.pdf'
        doc.metadata['doc_id'] = f"resume_pdf_{uuid.uuid4().hex[:8]}"
    
    print(f"✅ Loaded {len(documents)} pages from resume PDF")
    return documents



def crawl_website(base_url, max_pages=10):
    """
    Crawls a website starting from the base_url and returns all discovered pages
    within the same domain, up to max_pages.
    
    Args:
        base_url: The starting URL to crawl
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of URLs discovered
    """
    # Parse the base URL to extract domain
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    
    # Initialize sets to track pages
    discovered_urls = set()
    urls_to_visit = {base_url}
    visited_urls = set()
    
    print(f"🕸️ Starting web crawler from {base_url}")
    
    while urls_to_visit and len(visited_urls) < max_pages:
        # Get next URL to visit
        current_url = urls_to_visit.pop()
        
        # Skip if already visited
        if current_url in visited_urls:
            continue
            
        print(f"  Crawling: {current_url}")
        
        try:
            # Fetch the webpage
            response = requests.get(current_url, timeout=10)
            visited_urls.add(current_url)
            
            # Skip if not successful
            if response.status_code != 200:
                print(f"  ⚠️ Failed to fetch {current_url}: Status code {response.status_code}")
                continue
                
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Add current URL to discovered set
            discovered_urls.add(current_url)
            
            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Create absolute URL
                full_url = urljoin(current_url, href)
                parsed_url = urlparse(full_url)
                
                # Only follow links on the same domain and not anchors
                if (parsed_url.netloc == base_domain or not parsed_url.netloc) and '#' not in full_url:
                    # Normalize URL
                    normalized_url = urljoin(full_url, '.')
                    
                    # Add to visit queue if not visited yet
                    if normalized_url not in visited_urls and normalized_url not in urls_to_visit:
                        urls_to_visit.add(normalized_url)
        
        except Exception as e:
            print(f"  ⚠️ Error crawling {current_url}: {e}")
    
    print(f"✅ Crawling complete. Discovered {len(discovered_urls)} pages.")
    return list(discovered_urls)
  

def load_website_content():
    """Load content from the resume website using automatic crawling"""
    try:
        base_url = RESUME_URL
        print(f"🌐 Starting automatic website crawling from {base_url}")
        
        # Get all URLs from the website
        all_urls = crawl_website(base_url, max_pages=15)
        
        if not all_urls:
            print("⚠️ No URLs found during crawling. Falling back to base URL only.")
            all_urls = [base_url]
        
        print(f"🔍 Found {len(all_urls)} pages to process: {all_urls}")
        
        # Use UnstructuredURLLoader to fetch the website content
        loader = UnstructuredURLLoader(urls=all_urls)
        documents = loader.load()
        
        # Add metadata to identify website content
        for i, doc in enumerate(documents):
            doc.metadata['source_type'] = 'website'
            doc.metadata['url'] = all_urls[i % len(all_urls)]  # Map back to source URL
            doc.metadata['doc_id'] = f"website_{i}_{uuid.uuid4().hex[:8]}"
        
        print(f"✅ Loaded {len(documents)} documents from website")
        return documents
    except Exception as e:
        print(f"⚠️ Error loading website content: {e}")
        return []

def process_documents(documents: List[Document]):
    """Process documents with appropriate chunking based on source"""
    processed_docs = []
    
    pdf_docs = [doc for doc in documents if doc.metadata.get('source_type') == 'resume_pdf']
    website_docs = [doc for doc in documents if doc.metadata.get('source_type') == 'website']
    
    # For PDF resume - use larger chunks to preserve context
    if pdf_docs:
        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks for resume
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        pdf_chunks = pdf_splitter.split_documents(pdf_docs)
        print(f"📄 Split {len(pdf_docs)} PDF documents into {len(pdf_chunks)} chunks")
        
        # Label PDF chunks with section information
        for chunk in pdf_chunks:
            content = chunk.page_content.lower()
            if any(keyword in content for keyword in ["education", "university", "degree"]):
                chunk.metadata['section'] = "education"
            elif any(keyword in content for keyword in ["experience", "work", "job"]):
                chunk.metadata['section'] = "experience"
            elif any(keyword in content for keyword in ["skill", "technology", "framework"]):
                chunk.metadata['section'] = "skills"
            elif any(keyword in content for keyword in ["project", "portfolio"]):
                chunk.metadata['section'] = "projects"
            else:
                chunk.metadata['section'] = "general"
                
            processed_docs.append(chunk)
    
    # For website content - more granular chunking
    if website_docs:
        web_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly smaller for website content
            chunk_overlap=150,
            separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        web_chunks = web_splitter.split_documents(website_docs)
        print(f"🌐 Split {len(website_docs)} website documents into {len(web_chunks)} chunks")
        
        # Try to identify sections from HTML context
        for chunk in web_chunks:
            content = chunk.page_content.lower()
            if any(keyword in content for keyword in ["education", "university", "degree"]):
                chunk.metadata['section'] = "education"
            elif any(keyword in content for keyword in ["experience", "work", "job"]):
                chunk.metadata['section'] = "experience"
            elif any(keyword in content for keyword in ["skill", "technology", "framework"]):
                chunk.metadata['section'] = "skills"
            elif any(keyword in content for keyword in ["project", "portfolio"]):
                chunk.metadata['section'] = "projects"
            elif any(keyword in content for keyword in ["about", "profile", "summary"]):
                chunk.metadata['section'] = "about"
            else:
                chunk.metadata['section'] = "general"
                
            processed_docs.append(chunk)
    
    # Print some sample information
    if processed_docs:
        print(f"\n✅ Total processed chunks: {len(processed_docs)}")
        print("Sample chunk information:")
        print(f"  Section: {processed_docs[0].metadata.get('section', 'unknown')}")
        print(f"  Source: {processed_docs[0].metadata.get('source_type', 'unknown')}")
        print(f"  Content (first 100 chars): {processed_docs[0].page_content[:100]}...")
    
    return processed_docs

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
    
    # Create and persist the vector store
    try:
        print(f"💾 Storing {len(chunks)} chunks in ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        vector_store.persist()
        print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
        return vector_store
    except Exception as e:
        print(f"❌ Error storing documents in ChromaDB: {e}")
        raise

def test_retrieval(vector_store, queries=None):
    """Test retrieval from the vector store with sample queries"""
    if queries is None:
        queries = [
            "What is Ntwari Mike's education background?",
            "What programming languages does Ntwari know?",
            "What projects has Ntwari worked on?",
            "What are Ntwari's technical skills?"
        ]
    
    print("\n🔍 Testing document retrieval with sample queries:")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.similarity_search_with_relevance_scores(query, k=2)
        
        print(f"Retrieved {len(results)} results:")
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.4f}):")
            print(f"Section: {doc.metadata.get('section', 'Unknown')}")
            print(f"Source: {doc.metadata.get('source_type', 'Unknown')}")
            print(f"Content: {doc.page_content[:150]}...")

def main():
    """Run the complete document ingestion pipeline"""
    print("🤖 Starting resume chatbot data ingestion pipeline...")
    
    # Load documents from different sources
    pdf_docs = load_pdf_document()
    website_docs = load_website_content()
    
    # Combine all documents
    all_docs = pdf_docs + website_docs
    
    if not all_docs:
        print("❌ No documents loaded. Please check your file paths and website URLs.")
        return
    
    # Process documents into chunks
    chunks = process_documents(all_docs)
    
    # Store chunks in vector database
    vector_store = store_in_chromadb(chunks)
    
    # Test retrieval
    test_retrieval(vector_store)
    
    print("\n✅ Document ingestion completed successfully!")
    print(f"Vector database stored at: {CHROMA_PATH}")

if __name__ == "__main__":
    main()
