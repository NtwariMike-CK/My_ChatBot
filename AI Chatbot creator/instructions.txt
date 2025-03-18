# AI Chatbot Creator - Setup Instructions

This document provides instructions for setting up and running the AI Chatbot Creator system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 8GB+ RAM recommended
- At least 2GB of free disk space

## Setup Options

### Option 1: Docker (Recommended)

1. Install Docker on your system if you haven't already.
2. Clone or download this repository to your local machine.
3. Open a terminal and navigate to the project directory.
4. Build the Docker image:
   ```
   docker build -t ai-chatbot-creator .
   ```
5. Run the container:
   ```
   docker run -p 5000:5000 -v $(pwd)/data:/app/data ai-chatbot-creator
   ```
6. Access the application at http://localhost:5000

### Option 2: Manual Setup

1. Clone or download this repository to your local machine.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   flask run
   ```
6. Access the application at http://localhost:5000

## Project Structure

- `chatbot_engine.py` - Core logic for document processing and Q&A
- `app.py` - Flask web application
- `templates/` - HTML templates
- `data/` - Storage for document embeddings and metadata
- `uploads/` - Temporary storage for uploaded files

## Usage Instructions

1. **Adding Documents**:
   - Upload PDF, TXT, or MD files
   - Add web URLs for scraping
   - Paste text directly

2. **Asking Questions**:
   - Type your question in the chat interface
   - The system will retrieve relevant information and generate a response
   - All responses are based solely on the provided documents

3. **Supported Languages**:
   - The system supports queries and documents in multiple languages
   - Language detection is automatic

## Customization

### Model Selection

You can modify the embedding model in `chatbot_engine.py`:

```python
self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
```

Replace with other SentenceTransformer models as needed.

### System Parameters

You can adjust chunking parameters in `chunk_document()`:

```python
def chunk_document(self, document: Document, chunk_size: int = 1000, overlap: int = 200)
```

## Scaling Considerations

- For larger deployments, consider:
  - Using PostgreSQL with pgvector for vector storage
  - Implementing a task queue with Celery for document processing
  - Setting up a dedicated model server with optimized inference

## Troubleshooting

- **Memory Issues**: Reduce embedding dimensions or batch size
- **Slow Processing**: Adjust chunk size or use a lighter embedding model
- **Quality Issues**: Increase retrieval top_k or adjust LLM parameters

## Support

For issues or questions, please submit an issue in the repository.
