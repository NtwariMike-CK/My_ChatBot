import os
from typing import List, Dict, Any
import time

# Vector database and embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM components
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize directory paths
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()  # In Colab, use the current working directory

DATA_PATH = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Check if vector database exists
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError(f"❌ Vector database not found at {CHROMA_PATH}. Please run the ingest script first.")

# Define a resume-specific prompt template
RESUME_PROMPT_TEMPLATE = """
You are an AI assistant for Ntwari Mike Chris Kevin. Answer questions about Ntwari's resume, skills, education, 
experience, and projects using only the information from the provided context.

If you cannot find the specific information in the context, state that clearly rather than inventing details.
Always respond in a professional manner as if you are representing Ntwari Mike in a job application context.

Context:
{context}

Question: {question}

Answer:
"""

def get_embeddings():
    """Initialize and return the embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def initialize_llm():
    """Initialize the language model for answering questions"""
    try:
        # Using a smaller model that's suitable for this task
        model_name = "MBZUAI/LaMini-Flan-T5-783M"
        print(f"🔄 Loading language model: {model_name}")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a text generation pipeline with appropriate parameters
        text_generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=False,  # Deterministic outputs
            temperature=0.1,   # Low temperature for factual responses
        )
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        print(f"✅ Language model loaded successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing language model: {e}")
        raise

def retrieve_resume_context(query: str, db: Any, k: int = 3):
    """Retrieve relevant resume information based on the query"""
    # Get relevant chunks with similarity scores
    results = db.similarity_search_with_relevance_scores(query, k=k)
    
    if not results:
        return [], []
    
    # Extract documents and their metadata
    retrieved_docs = []
    sources = []
    
    for doc, score in results:
        # Collect document content
        retrieved_docs.append(doc)
        
        # Track source info
        source_type = doc.metadata.get('source_type', 'Unknown')
        section = doc.metadata.get('section', 'Unknown')
        sources.append(f"{source_type} ({section})")
        
        # Debug info
        print(f"📄 Found relevant context (score: {score:.4f}):")
        print(f"  Source: {source_type}, Section: {section}")
        print(f"  Content: {doc.page_content[:100]}...")
    
    return retrieved_docs, sources

def process_resume_query(query: str, db: Any, llm: Any) -> str:
    """Process a query about the resume and generate a response"""
    print(f"\n🔍 Processing query: '{query}'")
    
    # Get relevant documents
    docs, sources = retrieve_resume_context(query, db, k=4)
    
    if not docs:
        return "I don't have enough information about Ntwari Mike's resume to answer that question."
    
    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # Format the prompt with the resume-specific template
    prompt_template = ChatPromptTemplate.from_template(RESUME_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    
    # Generate response
    try:
        start_time = time.time()
        response_text = llm.predict(prompt)
        processing_time = time.time() - start_time
        print(f"⏱️ Response generated in {processing_time:.2f} seconds")
        
        return response_text.strip()
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I'm having trouble processing information about Ntwari Mike's resume right now."

def main():
    """Run the resume chatbot"""
    print("🤖 Starting Ntwari Mike's Resume Chatbot...")
    
    # Initialize embeddings and vector store
    embeddings = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Initialize language model
    llm = initialize_llm()
    
    print("\n✅ Resume Chatbot is ready! Type 'exit' to quit.\n")
    print("Ask questions about Ntwari Mike's skills, education, experience, or projects.")
    
    while True:
        query = input("\nYou: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Thank you for your interest in Ntwari Mike's resume!")
            break
        
        response = process_resume_query(query, db, llm)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main()
