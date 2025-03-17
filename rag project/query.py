import argparse
import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

# Set the ChromaDB storage path


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
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Ensure the path exists or create it
os.makedirs(CHROMA_PATH, exist_ok=True)

print(f"✅ Database path set to: {CHROMA_PATH}")


# Define the prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize HuggingFace Embeddings
embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Better performance for English content
                                            
# Load the language model (HuggingFace Pipeline)
text_generator = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-783M")
llm = HuggingFacePipeline(pipeline=text_generator)

# Load the ChromaDB vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def main(query_text):
    # Create CLI
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text


    # Search for similar documents
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0:
        print(f"No matching documents found.")
        return "I couldn't find any relevant information to answer your question."
        
    # Use a lower threshold or dynamic threshold based on the best match
    best_score = results[0][1]
    threshold = 0.5  # Lower base threshold
    
    # Dynamic threshold adjustment - if best result is very good, be more selective
    if best_score > 0.8:
        threshold = 0.6
    
    # Filter results
    filtered_results = [(doc, score) for doc, score in results if score >= threshold]
    
    if not filtered_results:
        print(f"Retrieved documents with scores: {[score for _, score in results]}")
        print(f"No documents met the similarity threshold of {threshold}")
        return "I found some information, but it might not be relevant to your question."

    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    
    # Include scores in debug output
    print(f"Using {len(filtered_results)} documents with scores: {[score for _, score in filtered_results]}")
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response using HuggingFace model
    response_text = llm.predict(prompt)

    # Extract sources if available
    sources = [doc.metadata.get("source", None) for doc, _score in filtered_results]
    return f"{response_text}\n\nSources: {sources}"
    
if __name__ == "__main__":
    while True:
      """Interactive chatbot that allows users to type messages continuously."""
      print("🤖 RAG Chatbot is running! Type 'exit' to quit.\n")
      query_text = input("You: ")  # Get user input
      if query_text.lower() == "exit":  # Exit condition
        print("👋 Goodbye!")
        break
      print(f"Bot: {main(query_text)}")
