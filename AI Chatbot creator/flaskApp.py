# app.py - Flask Web Application for the AI Chatbot Creator
from flask import Flask, request, render_template, jsonify, session
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename

# Import our chatbot engine
from chatbot_engine import ChatbotEngine, Document

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize chatbot engine
chatbot = ChatbotEngine()

@app.route('/')
def index():
    # Create a session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document uploads."""
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                doc_id = chatbot.add_document(filename, content=f.read())
            return jsonify({'success': True, 'document_id': doc_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        finally:
            # Clean up file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    elif 'url' in request.form:
        url = request.form['url']
        try:
            doc_id = chatbot.add_document(url)
            return jsonify({'success': True, 'document_id': doc_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    elif 'text' in request.form:
        text = request.form['text']
        source = request.form.get('source', 'text_input')
        try:
            doc_id = chatbot.add_document(source, text_content=text)
            return jsonify({'success': True, 'document_id': doc_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': 'No document provided'}), 400

@app.route('/query', methods=['POST'])
def query():
    """Process a question and return the response."""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
        
    question = data['question']
    try:
        response = chatbot.query(question)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
