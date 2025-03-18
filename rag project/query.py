import os
import time
from typing import List, Dict, Tuple, Any

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

# Define the prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}

Answer:
"""

# Initialize embedding model
def get_embeddings():
    """Initialize and return the embedding model with normalization"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Ensure normalization for consistent scores
    )
    return embeddings

# Initialize language model
def initialize_llm():
    """Initialize and return the language model for answering questions"""
    try:
        # Load the model with appropriate settings
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
        model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
        
        # Configure the generation pipeline with better parameters
        text_generator = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        return llm
    except Exception as e:
        print(f"❌ Error initializing language model: {e}")
        raise

# Process a query and return a response
def process_query(query_text: str, db: Any, llm: Any, threshold: float = 0.3, k: int = 5) -> str:
    """
    Process a query and generate a response based on relevant documents.
    
    Args:
        query_text: The user's query
        db: The ChromaDB vector store
        llm: The language model
        threshold: Similarity threshold (documents below this won't be used)
        k: Number of documents to retrieve
    
    Returns:
        str: The generated response
    """
    print(f"\n🔍 Processing query: '{query_text}'")
    
    # Adjust k if needed based on collection size
    collection_size = db._collection.count()
    if k > collection_size:
        print(f"⚠️ Adjusting k from {k} to {collection_size} (collection size)")
        k = max(1, collection_size)
    
    # Search for similar documents
    start_time = time.time()
    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=k)
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return "I encountered a problem while searching for relevant information."
    
    query_time = time.time() - start_time
    print(f"⏱️ Query took {query_time:.2f} seconds")
    
    if not results:
        print("❌ No matching documents found.")
        return "I couldn't find any relevant information to answer your question."
    
    # Print raw scores for debugging
    print(f"📊 Raw relevance scores: {[f'{score:.4f}' for _, score in results]}")
    
    # Normalize scores if they're negative
    min_score = min(score for _, score in results)
    if min_score < 0:
        print("⚠️ Detected negative scores. Normalizing to 0-1 range.")
        # Convert negative scores to positive range (0-1)
        results = [(doc, 1 - (abs(score) / max(1, abs(min_score)))) for doc, score in results]
        print(f"📊 Normalized scores: {[f'{score:.4f}' for _, score in results]}")
    
    # Apply threshold to filter results
    filtered_results = [(doc, score) for doc, score in results if score >= threshold]
    
    if not filtered_results:
        print(f"⚠️ No documents met the similarity threshold of {threshold}")
        
        # Fallback: use the top result regardless of score if nothing passes threshold
        top_result = [(results[0][0], results[0][1])]
        print(f"🔄 Fallback: Using top result with score {results[0][1]:.4f}")
        filtered_results = top_result
    
    # Prepare context from retrieved documents
    context_docs = []
    for i, (doc, score) in enumerate(filtered_results):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        context_docs.append(f"[Document {i+1} (Relevance: {score:.2f}) from {source}]\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(context_docs)
    
    print(f"✅ Using {len(filtered_results)} documents with scores: {[f'{score:.4f}' for _, score in filtered_results]}")
    
    # Format the prompt for the language model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate response using the language model
    try:
        response_text = llm.predict(prompt)
        
        # Format the response with sources
        sources = [doc.metadata.get('source', 'Unknown').split('/')[-1] for doc, _ in filtered_results]
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        
        return f"{response_text.strip()}\n\nSources: {', '.join(unique_sources)}"
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I encountered a problem while generating a response."

# Main function to run the chatbot
def main():
    """Run the interactive chatbot"""
    print("⚙️ Initializing RAG chatbot components...")
    
    # Initialize embeddings and vector store
    embeddings = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Initialize language model
    llm = initialize_llm()
    
    print("🤖 RAG Chatbot is running! Type 'exit' to quit.\n")
    
    while True:
        query_text = input("You: ")
        
        if query_text.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        
        response = process_query(query_text, db, llm, threshold=0.25)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
