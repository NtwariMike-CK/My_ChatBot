# Import necessary libraries
import os
import re
import time
import json
import hashlib
import requests
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Third-party imports
import nltk
import faiss
import PyPDF2
import tiktoken
import markdown
import langdetect
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Data structures
@dataclass
class Document:
    """Represents a document with its metadata and content."""
    id: str
    source: str  # URL, file path, etc.
    content: str
    metadata: Dict[str, Any] = None
    page_count: int = 1
    language: str = "en"
    
@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    id: str
    doc_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """Initialize with embedding model."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_from_url(self, url: str) -> Document:
        """Extract content from a web URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator='\n', strip=True)
            
            # Count approximate pages (5000 chars per page)
            page_count = max(1, len(text) // 5000)
            
            # Detect language
            try:
                language = langdetect.detect(text[:1000])
            except:
                language = "en"
                
            doc_id = hashlib.md5(url.encode()).hexdigest()
            
            return Document(
                id=doc_id,
                source=url,
                content=text,
                page_count=page_count,
                language=language,
                metadata={"title": soup.title.string if soup.title else url}
            )
        except Exception as e:
            raise ValueError(f"Failed to extract content from URL: {e}")
    
    def extract_from_pdf(self, file_content: bytes) -> Document:
        """Extract content from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
                
            page_count = len(pdf_reader.pages)
            
            # Detect language
            try:
                language = langdetect.detect(text[:1000])
            except:
                language = "en"
                
            doc_id = hashlib.md5(file_content).hexdigest()
            
            return Document(
                id=doc_id,
                source="pdf_upload",
                content=text,
                page_count=page_count,
                language=language,
                metadata={"title": f"PDF Document ({page_count} pages)"}
            )
        except Exception as e:
            raise ValueError(f"Failed to extract content from PDF: {e}")
    
    def extract_from_text(self, text: str, source: str = "text_upload") -> Document:
        """Process plain text or markdown content."""
        if source.endswith('.md'):
            # Convert markdown to plain text
            text = markdown.markdown(text)
            text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
        
        # Count approximate pages (5000 chars per page)
        page_count = max(1, len(text) // 5000)
        
        # Detect language
        try:
            language = langdetect.detect(text[:1000])
        except:
            language = "en"
            
        doc_id = hashlib.md5(text.encode()).hexdigest()
        
        return Document(
            id=doc_id,
            source=source,
            content=text,
            page_count=page_count,
            language=language,
            metadata={"title": f"Text Document ({page_count} pages)"}
        )
    
    def chunk_document(self, document: Document, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
        """Split document into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(document.content)
        chunks = []
        
        # For small documents, create a single chunk
        if document.page_count <= 5:
            chunk_id = f"{document.id}_chunk_0"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=document.id,
                content=document.content,
                metadata={
                    "source": document.source,
                    "language": document.language,
                    **document.metadata if document.metadata else {}
                }
            ))
            return chunks
        
        # For larger documents, create multiple chunks
        i = 0
        while i < len(tokens):
            # Get chunk tokens with overlap
            end = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            
            # Convert back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk
            chunk_id = f"{document.id}_chunk_{len(chunks)}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=document.id,
                content=chunk_text,
                metadata={
                    "source": document.source,
                    "chunk_index": len(chunks),
                    "language": document.language,
                    **document.metadata if document.metadata else {}
                }
            ))
            
            # Move to next chunk, accounting for overlap
            i += (chunk_size - overlap)
            
        return chunks
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for a list of chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            
        return chunks


class VectorStore:
    """Simple vector store using FAISS."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize the vector store."""
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = {}
        self.documents = {}
    
    def add_document(self, document: Document, processor: DocumentProcessor):
        """Add a document to the vector store."""
        # Store document
        self.documents[document.id] = document
        
        # Process chunks
        chunks = processor.chunk_document(document)
        chunks = processor.generate_embeddings(chunks)
        
        # Add chunks to store and index
        chunk_vectors = []
        chunk_ids = []
        
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            chunk_vectors.append(chunk.embedding)
            chunk_ids.append(chunk.id)
            
        if chunk_vectors:
            # Add to FAISS index
            chunk_vectors = np.array(chunk_vectors).astype('float32')
            faiss.normalize_L2(chunk_vectors)  # Normalize for cosine similarity
            
            current_index = self.index.ntotal
            self.index.add(chunk_vectors)
            
            for i, chunk_id in enumerate(chunk_ids):
                chunk = self.chunks[chunk_id]
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["faiss_id"] = current_index + i
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Chunk]:
        """Search for similar chunks using a query vector."""
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        D, I = self.index.search(query_vector, top_k)
        
        # Get chunks
        results = []
        for i, faiss_id in enumerate(I[0]):
            if faiss_id == -1 or D[0][i] == float('inf'):  # Invalid result
                continue
                
            # Find chunk with this faiss_id
            for chunk_id, chunk in self.chunks.items():
                if chunk.metadata and chunk.metadata.get("faiss_id") == faiss_id:
                    results.append((chunk, float(D[0][i])))
                    break
        
        return [r[0] for r in sorted(results, key=lambda x: x[1])]


class ChatbotEngine:
    """Main chatbot engine that integrates all components."""
    
    def __init__(self):
        """Initialize the chatbot engine."""
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Load a lightweight LLM
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    
    def add_document(self, source: str, content: bytes = None, text_content: str = None) -> str:
        """Add a document to the knowledge base."""
        document = None
        
        # Process based on source type
        if source.startswith('http'):
            document = self.document_processor.extract_from_url(source)
        elif source.endswith('.pdf') and content:
            document = self.document_processor.extract_from_pdf(content)
        elif content:
            # Try to decode content to string
            try:
                text = content.decode('utf-8')
                document = self.document_processor.extract_from_text(text, source)
            except:
                raise ValueError("Unsupported document format")
        elif text_content:
            document = self.document_processor.extract_from_text(text_content, source)
        else:
            raise ValueError("No document content provided")
            
        # Add to vector store
        self.vector_store.add_document(document, self.document_processor)
        return document.id
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Process a query and generate a response."""
        # Detect language
        try:
            language = langdetect.detect(question)
        except:
            language = "en"
            
        # Generate query embedding
        query_embedding = self.document_processor.embedding_model.encode(question)
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(query_embedding, top_k)
        
        if not relevant_chunks:
            if language == "en":
                return "I don't have enough information to answer that question based on the provided documents."
            else:
                # Return a basic multilingual response
                responses = {
                    "es": "No tengo suficiente información para responder esta pregunta basada en los documentos proporcionados.",
                    "fr": "Je n'ai pas assez d'informations pour répondre à cette question sur la base des documents fournis.",
                    "de": "Ich habe nicht genügend Informationen, um diese Frage anhand der bereitgestellten Dokumente zu beantworten.",
                    "it": "Non ho abbastanza informazioni per rispondere a questa domanda in base ai documenti forniti.",
                    "zh": "根据提供的文档，我没有足够的信息来回答这个问题。",
                    "ja": "提供された文書に基づいて、その質問に答えるための十分な情報がありません。"
                }
                return responses.get(language, "I don't have enough information to answer that question based on the provided documents.")
        
        # Prepare context from chunks
        context = "\n\n---\n\n".join([chunk.content for chunk in relevant_chunks])
        
        # Prepare prompt for the LLM
        prompt = f"""You are an AI assistant that only answers questions based on the provided information. 
If you don't know the answer based on the provided context, say so.

Context information:
{context}

Question: {question}

Answer:"""

        # Use LLM to generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=512, 
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        
        return response


# Example usage function
def run_chatbot_demo():
    """Run a demo of the chatbot."""
    chatbot = ChatbotEngine()
    
    # Add documents
    print("Adding sample documents...")
    
    # Example text about AI
    ai_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
    """
    
    chatbot.add_document("ai_info.txt", text_content=ai_text)
    
    # Example text about Python
    python_text = """
    Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    It is often described as a "batteries included" language due to its comprehensive standard library.
    Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.
    Python 2.0 was released in 2000 and introduced new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support.
    Python 3.0, released in 2008, was a major revision that is not completely backward-compatible with earlier versions.
    Python consistently ranks as one of the most popular programming languages.
    """
    
    chatbot.add_document("python_info.txt", text_content=python_text)
    
    print("Documents added to the knowledge base.")
    print("\nYou can now ask questions about AI and Python. Type 'exit' to quit.\n")
    
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
            
        start_time = time.time()
        response = chatbot.query(question)
        end_time = time.time()
        
        print(f"\nResponse (generated in {end_time - start_time:.2f}s):")
        print(response)
        print("\n" + "-" * 50 + "\n")


# Entry point
if __name__ == "__main__":
    run_chatbot_demo()
