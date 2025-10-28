#!/usr/bin/env python3
"""
Web API for Medical RAG System using Flask
Allows deployment via ngrok for remote access
"""
import os
os.environ['USE_TORCH'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from train import MedicalRAG

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize RAG system
print(">> Initializing Medical RAG System...")
base_dir = Path(__file__).parent.parent
embeddings_dir = base_dir / 'Embeddings'

try:
    rag = MedicalRAG(
        embeddings_dir=str(embeddings_dir),
        model_type='demo'
    )
    print(">> RAG system initialized successfully!")
except Exception as e:
    print(f"ERROR: Error initializing RAG: {e}")
    rag = None

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical RAG System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        .stat-card h3 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 5px;
        }
        .stat-card p {
            color: #6c757d;
            font-size: 0.9em;
        }
        .search-box {
            margin-bottom: 20px;
        }
        .search-box textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        .search-box textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #e9ecef;
            color: #495057;
        }
        .btn-secondary:hover {
            background: #dee2e6;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .result {
            display: none;
            margin-top: 20px;
        }
        .answer-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }
        .answer-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .answer-text {
            line-height: 1.6;
            color: #333;
        }
        .sources {
            margin-top: 20px;
        }
        .source-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid #e9ecef;
        }
        .source-card .source-title {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .source-card .similarity {
            color: #28a745;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        .source-card .text {
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .examples {
            margin-top: 20px;
        }
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            background: #e9ecef;
            border-radius: 20px;
            margin: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical RAG System</h1>
            <p>Ask medical questions and get AI-powered answers from 9 medical textbooks</p>
        </div>
        
        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h3 id="total-chunks">-</h3>
                    <p>Knowledge Chunks</p>
                </div>
                <div class="stat-card">
                    <h3 id="total-books">-</h3>
                    <p>Medical Textbooks</p>
                </div>
                <div class="stat-card">
                    <h3>384</h3>
                    <p>Vector Dimensions</p>
                </div>
            </div>
            
            <div class="search-box">
                <textarea id="question" placeholder="Ask a medical question... (e.g., What is diabetes mellitus?)"></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="askQuestion()">🔍 Search</button>
                <button class="btn-secondary" onclick="clearResults()">🗑️ Clear</button>
            </div>
            
            <div class="examples">
                <strong>Example questions:</strong><br>
                <span class="example-btn" onclick="setQuestion('What is diabetic ketoacidosis?')">Diabetic ketoacidosis</span>
                <span class="example-btn" onclick="setQuestion('What are the symptoms of heart failure?')">Heart failure symptoms</span>
                <span class="example-btn" onclick="setQuestion('How is hypertension treated?')">Hypertension treatment</span>
                <span class="example-btn" onclick="setQuestion('What causes pneumonia?')">Pneumonia causes</span>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Searching medical knowledge base...</p>
            </div>
            
            <div class="result" id="result">
                <div class="answer-box">
                    <h3>💡 Answer</h3>
                    <div class="answer-text" id="answer"></div>
                </div>
                
                <div class="sources">
                    <h3>📚 Sources</h3>
                    <div id="sources"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>⚠️ For educational purposes only. Not a substitute for professional medical advice.</p>
            <p>Powered by Medical RAG System | 9 Medical Textbooks | 43,258+ Medical Facts</p>
        </div>
    </div>
    
    <script>
        // Load stats on page load
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('total-chunks').textContent = data.total_chunks.toLocaleString();
                document.getElementById('total-books').textContent = data.total_books;
            });
        
        function setQuestion(text) {
            document.getElementById('question').value = text;
        }
        
        function clearResults() {
            document.getElementById('question').value = '';
            document.getElementById('result').style.display = 'none';
        }
        
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question, top_k: 3})
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Show answer
                document.getElementById('answer').textContent = data.answer;
                
                // Show sources
                const sourcesHtml = data.sources.map((source, i) => `
                    <div class="source-card">
                        <div class="source-title">${i+1}. ${source.book}</div>
                        <div class="similarity">Similarity: ${(source.similarity * 100).toFixed(1)}%</div>
                        <div class="text">${source.text.substring(0, 300)}...</div>
                    </div>
                `).join('');
                document.getElementById('sources').innerHTML = sourcesHtml;
                
                // Show result
                document.getElementById('result').style.display = 'block';
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error.message);
            }
        }
        
        // Allow Enter key to search (Ctrl+Enter for newline)
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    total_chunks = sum(len(data['metadata']) for data in rag.embeddings_data.values())
    
    return jsonify({
        'total_chunks': total_chunks,
        'total_books': len(rag.embeddings_data),
        'status': 'ready'
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer a medical question."""
    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        question = data.get('question', '')
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get answer
        result = rag.answer_question(question, top_k=top_k)
        
        # Format response
        response = {
            'question': question,
            'answer': result['answer'],
            'sources': [
                {
                    'book': s['book'],
                    'similarity': s['similarity'],
                    'text': s['text']
                }
                for s in result['sources'][:top_k]
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search medical knowledge base."""
    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Search
        results = rag.search(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if rag is not None else 'unhealthy',
        'message': 'Medical RAG System is running'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print(">> Medical RAG Web Server")
    print("="*70)
    print("\n>> Local URL: http://localhost:5000")
    print(">> Network URL: http://0.0.0.0:5000")
    print("\n>> To expose via ngrok:")
    print("   ngrok http 5000")
    print("\n>> Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
