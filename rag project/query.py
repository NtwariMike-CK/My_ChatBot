import argparse
import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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

# DATA_PATH = os.path.abspath("Final chatbot/rag project/docs/")

# Initialize ChromaDB client (Persistent storage) the location to store the db
# Define the path to the database folder dynamically

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Ensure the path exists or create it
os.makedirs(CHROMA_PATH, exist_ok=True)

print(f"✅ Database path set to: {CHROMA_PATH}")


# CHROMA_PATH = "chroma"
CHROMA_PATH = os.path.abspath("Final chatbot/rag project/chroma_db")
# Define the prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize HuggingFace Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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

    # Load the ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for similar documents
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print(f"Unable No matching found.")
        return
    elif results[0][1] < 0.7:
        print(f"The results are less similar hence no matching")
        return


    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response using HuggingFace model
    response_text = llm.predict(prompt)

    # Extract sources if available
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    while True:
      """Interactive chatbot that allows users to type messages continuously."""
      print("🤖 RAG Chatbot is running! Type 'exit' to quit.\n")
      query_text = input("You: ")  # Get user input
      if query_text.lower() == "exit":  # Exit condition
        print("👋 Goodbye!")
        break
      print(f"Bot: {main(query_text)}")
