import os
import time
from typing import List, Dict, Tuple, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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
You are a helpful assistant answering questions about a resume or document. 
Use ONLY the information provided in the context below to answer the question.
If the information is not present in the context, say "Based on the available information, I cannot answer this question."

Context:
{context}

Question: {question}

Instructions:
1. Answer precisely based on the context
2. Do not make up information not in the context
3. If multiple pieces of information are relevant, organize them clearly
4. For questions about dates, skills, or experience, cite specific sections

Answer:
"""

# Initialize embedding model
def get_embeddings():
    """Initialize and return the embedding model with normalization"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Better for Q&A tasks
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Ensure normalization
    )
    return embeddings

# Initialize language model
def initialize_llm():
    """Initialize and return the language model for answering questions"""
    try:
        # Use Flan-T5 for better question answering
        model_name = "google/flan-t5-base"
        print(f"🔄 Loading language model: {model_name}")
        
        # Load the model with appropriate settings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Configure the generation pipeline with better parameters
        text_generator = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.4,  # Lower temperature for more factual responses
            top_p=0.9,
            top_k=50,
        )
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        print(f"✅ Language model loaded successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing language model: {e}")
        raise

# Process user queries through hybrid search
def hybrid_search(query: str, db: Any, k: int = 5) -> List[Tuple[Any, float]]:
    """
    Perform hybrid search combining semantic search and MMR for better results.
    
    Args:
        query: The user's query
        db: ChromaDB vector store
        k: Number of results to return
        
    Returns:
        List of (document, score) tuples
    """
    # Perform semantic search
    semantic_results = db.similarity_search_with_relevance_scores(query, k=k)
    
    # Perform MMR search for diversity
    try:
        mmr_results = db.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
        # Convert MMR results to same format as semantic results (with default scores)
        mmr_results_with_scores = [(doc, 0.65) for doc in mmr_results]  # Default score for MMR
    except Exception as e:
        print(f"⚠️ MMR search failed: {e}")
        mmr_results_with_scores = []
    
    # Combine results with a weighted approach
    combined_results = []
    seen_ids = set()
    
    # Add semantic results first
    for doc, score in semantic_results:
        doc_id = doc.metadata.get('doc_id', doc.page_content[:50])
        if doc_id not in seen_ids:
            combined_results.append((doc, score))
            seen_ids.add(doc_id)
    
    # Add MMR results that are not duplicates
    for doc, score in mmr_results_with_scores:
        doc_id = doc.metadata.get('doc_id', doc.page_content[:50])
        if doc_id not in seen_ids:
            combined_results.append((doc, score))
            seen_ids.add(doc_id)
    
    # Sort by score and take top k
    return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]

# Analyze the query to determine the most relevant sections
def analyze_query(query: str) -> List[str]:
    """
    Analyze the query to determine which sections are most relevant.
    
    Args:
        query: The user's query
        
    Returns:
        List of relevant section names
    """
    query_lower = query.lower()
    relevant_sections = []
    
    # Education related queries
    if any(keyword in query_lower for keyword in ["education", "degree", "university", "college", "school", "study"]):
        relevant_sections.append("education")
    
    # Experience related queries
    if any(keyword in query_lower for keyword in ["experience", "work", "job", "position", "company", "career"]):
        relevant_sections.append("experience")
    
    # Skills related queries
    if any(keyword in query_lower for keyword in ["skill", "technology", "programming", "framework", "tool", "language"]):
        relevant_sections.append("skills")
    
    # If no specific sections identified, return empty list (all sections)
    return relevant_sections

# Advanced query processing with section filtering
def process_query(query_text: str, db: Any, llm: Any, threshold: float = 0.25, k: int = 5) -> str:
    """
    Process a query and generate a response based on relevant documents.
    
    Args:
        query_text: The user's query
        db: The ChromaDB vector store
        llm: The language model
        threshold: Similarity threshold
        k: Number of documents to retrieve
    
    Returns:
        str: The generated response
    """
    print(f"\n🔍 Processing query: '{query_text}'")
    
    # Analyze the query to determine relevant sections
    relevant_sections = analyze_query(query_text)
    if relevant_sections:
        print(f"📊 Query analysis: Relevant sections: {', '.join(relevant_sections)}")
    
    # Adjust k if needed based on collection size
    collection_size = db._collection.count()
    if k > collection_size:
        print(f"⚠️ Adjusting k from {k} to {collection_size} (collection size)")
        k = max(1, collection_size)
    
    # Search for similar documents
    start_time = time.time()
    try:
        # Use hybrid search for better results
        results = hybrid_search(query_text, db, k=k)
        
        # Filter by relevant sections if specified
        if relevant_sections:
            filtered_by_section = [
                (doc, score) for doc, score in results 
                if doc.metadata.get('section', 'general') in relevant_sections
            ]
            
            # If we have results after filtering, use them
            if filtered_by_section:
                results = filtered_by_section
                print(f"✅ Filtered results by sections: {', '.join(relevant_sections)}")
            else:
                print(f"⚠️ No results found in specified sections. Using all results.")
        
    except Exception as e:
        print(f"❌ Error during document retrieval: {e}")
        return "I encountered a problem while searching for relevant information."
    
    query_time = time.time() - start_time
    print(f"⏱️ Query took {query_time:.2f} seconds")
    
    if not results:
        print("❌ No matching documents found.")
        return "I couldn't find any relevant information to answer your question."
    
    # Print raw scores for debugging
    print(f"📊 Raw relevance scores: {[f'{score:.4f}' for _, score in results]}")
    
    # Apply threshold to filter results
    filtered_results = [(doc, score) for doc, score in results if score >= threshold]
    
    if not filtered_results:
        print(f"⚠️ No documents met the similarity threshold of {threshold}")
        
        # Fallback: use the top 2 results regardless of score
        top_results = [(results[0][0], results[0][1])]
        if len(results) > 1:
            top_results.append((results[1][0], results[1][1]))
            
        print(f"🔄 Fallback: Using top {len(top_results)} results")
        filtered_results = top_results
    
    # Prepare context from retrieved documents
    context_docs = []
    for i, (doc, score) in enumerate(filtered_results):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        section = doc.metadata.get('section', 'general')
        context_docs.append(f"[Document {i+1} | Source: {source} | Section: {section} | Relevance: {score:.2f}]\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(context_docs)
    
    print(f"✅ Using {len(filtered_results)} documents with scores: {[f'{score:.4f}' for _, score in filtered_results]}")
    
    # Format the prompt for the language model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate response using the language model
    try:
        start_time = time.time()
        response_text = llm.predict(prompt)
        processing_time = time.time() - start_time
        print(f"⏱️ Response generation took {processing_time:.2f} seconds")
        
        # Format the response with sources
        sources = [f"{doc.metadata.get('source', 'Unknown').split('/')[-1]} ({doc.metadata.get('section', 'general')})" 
                  for doc, _ in filtered_results]
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        
        return f"{response_text.strip()}\n\nSources: {', '.join(unique_sources)}"
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I encountered a problem while generating a response."

# Conversation history management
def maintain_conversation_history(history: List[Dict], query: str, response: str, max_history: int = 3) -> List[Dict]:
    """
    Maintain a conversation history for context.
    
    Args:
        history: Current conversation history
        query: The user's query
        response: The system's response
        max_history: Maximum number of turns to keep
        
    Returns:
        Updated conversation history
    """
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})
    
    # Keep only the last max_history turns
    if len(history) > max_history * 2:
        history = history[-max_history*2:]
    
    return history

# Main function to run the chatbot
def main():
    """Run the interactive chatbot"""
    print("⚙️ Initializing improved RAG chatbot...")
    
    # Initialize embeddings and vector store
    embeddings = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Initialize language model
    llm = initialize_llm()
    
    print("🤖 Improved RAG Chatbot is running! Type 'exit' to quit.\n")
    
    conversation_history = []
    
    while True:
        query_text = input("You: ")
        
        if query_text.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        
        response = process_query(query_text, db, llm, threshold=0.20, k=3)
        print(f"Bot: {response}")
        
        # Update conversation history
        conversation_history = maintain_conversation_history(conversation_history, query_text, response)

if __name__ == "__main__":
    main()
