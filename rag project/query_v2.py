import os
import time
from typing import List, Dict, Tuple, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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

# Ensure the paths exist
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError(f"❌ Vector database NOT found: {CHROMA_PATH}. Please run ingest.py first.")
print(f"✅ Vector database found at: {CHROMA_PATH}")

# Define a simplified prompt template that's easier for the model to process
PROMPT_TEMPLATE = """
Answer the question using only the information from the provided context.
If you can't find a specific answer in the context, summarize what's available that's most relevant.

Context:
{context}

Question: {question}

Answer:
"""

# Initialize embedding model
def get_embeddings():
    """Initialize and return the embedding model with normalization"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Reverting to a reliable model that works well
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

# Initialize language model
def initialize_llm():
    """Initialize and return the language model for answering questions"""
    try:
        # Use a model that's more reliable for simpler instruction following
        model_name = "MBZUAI/LaMini-Flan-T5-783M"  # Go back to your original model
        print(f"🔄 Loading language model: {model_name}")
        
        # Load the model with appropriate settings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Configure the generation pipeline with better parameters for extractive QA
        text_generator = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_length=512,
            do_sample=False,  # Set to False for more deterministic outputs
            temperature=0.1,   # Lower temperature for more focused answers
        )
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        print(f"✅ Language model loaded successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing language model: {e}")
        raise

# Enhanced document retrieval function
def retrieve_documents(query: str, db: Any, k: int = 5):
    """
    Retrieve relevant documents for a query
    
    Args:
        query: The user's query
        db: ChromaDB vector store
        k: Number of documents to retrieve
        
    Returns:
        List of (document, score) tuples
    """
    # Direct similarity search - simple but effective
    results = db.similarity_search_with_relevance_scores(query, k=k)
    
    # Log the results for debugging
    print(f"📄 Retrieved {len(results)} documents")
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        section = doc.metadata.get('section', 'general')
        print(f"  Doc {i+1}: {source} | {section} | Score: {score:.4f}")
        print(f"  Content: {doc.page_content[:100]}...")
    
    return results

# Simplified query processing
def process_query(query_text: str, db: Any, llm: Any, k: int = 3) -> str:
    """
    Process a query and generate a response based on relevant documents.
    
    Args:
        query_text: The user's query
        db: The ChromaDB vector store
        llm: The language model
        k: Number of documents to retrieve
    
    Returns:
        str: The generated response
    """
    print(f"\n🔍 Processing query: '{query_text}'")
    
    # Get relevant documents
    results = retrieve_documents(query_text, db, k=k)
    
    if not results:
        return "I couldn't find any relevant information to answer your question."
    
    # Prepare context from retrieved documents - SIMPLIFIED FORMAT
    context_parts = []
    for i, (doc, _) in enumerate(results):
        # Extract just the content without complex formatting
        context_parts.append(doc.page_content)
    
    # Join with simple separators
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Format the prompt with simplified template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Print the actual prompt for debugging
    print("\n🔍 DEBUG - Sending this prompt to the model:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Generate response using the language model
    try:
        start_time = time.time()
        response_text = llm.predict(prompt)
        processing_time = time.time() - start_time
        print(f"⏱️ Response generation took {processing_time:.2f} seconds")
        
        if not response_text or response_text.strip() == "":
            response_text = "I found some relevant information but couldn't formulate a specific answer."
        
        # Simply add the sources without complex formatting
        sources = [doc.metadata.get('source', 'Unknown').split('/')[-1] for doc, _ in results]
        unique_sources = list(dict.fromkeys(sources))
        
        return f"{response_text.strip()}\n\nSources: {', '.join(unique_sources)}"
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I encountered a problem while generating a response."

# Main function to run the chatbot
def main():
    """Run the interactive chatbot"""
    print("⚙️ Initializing fixed RAG chatbot...")
    
    # Initialize embeddings and vector store
    embeddings = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Initialize language model
    llm = initialize_llm()
    
    # Run a test query to verify the system works
    test_query = "What is this document about?"
    print("\n🧪 Running test query: " + test_query)
    test_response = process_query(test_query, db, llm, k=2)
    print(f"Test response: {test_response}")
    
    print("\n🤖 RAG Chatbot is running! Type 'exit' to quit.\n")
    
    while True:
        query_text = input("You: ")
        
        if query_text.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        
        response = process_query(query_text, db, llm, k=3)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
