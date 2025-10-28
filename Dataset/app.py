"""
Medical RAG Web Server with Evaluation API Support
Supports both web UI and evaluation endpoint format
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

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

# Simple HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical RAG System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
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
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 30px; }
        .search-box { margin-bottom: 20px; }
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
        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            margin-bottom: 20px;
        }
        button:hover { opacity: 0.9; }
        .results { margin-top: 20px; }
        .answer-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        .answer-box h3 { color: #667eea; margin-bottom: 10px; }
        .source {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid #e9ecef;
        }
        .source h4 { color: #764ba2; margin-bottom: 5px; font-size: 0.9em; }
        .source p { color: #6c757d; font-size: 0.9em; line-height: 1.6; }
        .loading { text-align: center; padding: 20px; color: #667eea; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical RAG System</h1>
            <p>AI-Powered Medical Knowledge Assistant</p>
        </div>
        <div class="content">
            <div class="search-box">
                <textarea id="question" placeholder="Ask a medical question..."></textarea>
            </div>
            <button onclick="askQuestion()">Ask Question</button>
            <div id="results" class="results"></div>
        </div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question) return alert('Please enter a question');
            
            document.getElementById('results').innerHTML = '<div class="loading">Processing...</div>';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, top_k: 3 })
                });
                
                const data = await response.json();
                
                let html = '<div class="answer-box"><h3>Answer:</h3><p>' + data.answer + '</p></div>';
                
                if (data.sources && data.sources.length > 0) {
                    html += '<h3>Sources:</h3>';
                    data.sources.forEach((source, i) => {
                        html += '<div class="source">';
                        html += '<h4>' + source.book + ' (Similarity: ' + (source.similarity * 100).toFixed(1) + '%)</h4>';
                        html += '<p>' + source.text + '</p>';
                        html += '</div>';
                    });
                }
                
                document.getElementById('results').innerHTML = html;
            } catch (error) {
                document.getElementById('results').innerHTML = '<div class="answer-box"><h3>Error:</h3><p>' + error + '</p></div>';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ask', methods=['GET'])
def ask_get():
    """Handle GET requests to /api/ask with helpful information."""
    return jsonify({
        'error': 'Method Not Allowed',
        'message': 'This endpoint only accepts POST requests',
        'usage': {
            'method': 'POST',
            'url': '/api/ask',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'query': 'Your medical question here',
                'top_k': 3
            },
            'example': {
                'query': 'What is the pregnancy category for acyclovir?',
                'top_k': 3
            }
        },
        'expected_response': {
            'answer': 'string',
            'contexts': ['string', '...']
        }
    }), 405

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Medical RAG System is running'})

@app.route('/api')
@app.route('/api/')
def api_docs():
    """API documentation."""
    return jsonify({
        'name': 'Medical RAG API',
        'version': '1.0',
        'description': 'AI-Powered Medical Knowledge Assistant',
        'endpoints': {
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint',
                'response': {'status': 'string', 'message': 'string'}
            },
            '/api/ask': {
                'method': 'POST',
                'description': 'Ask a medical question',
                'request': {
                    'query': 'string (required)',
                    'top_k': 'integer (optional, default: 3)'
                },
                'response': {
                    'answer': 'string',
                    'contexts': ['string', '...']
                },
                'example': {
                    'curl': 'curl -X POST http://localhost:5000/api/ask -H "Content-Type: application/json" -d \'{"query":"What is acyclovir?","top_k":3}\''
                }
            },
            '/api/search': {
                'method': 'POST',
                'description': 'Semantic search in medical knowledge base',
                'request': {
                    'query': 'string (required)',
                    'top_k': 'integer (optional, default: 5)'
                },
                'response': {
                    'query': 'string',
                    'results': []
                }
            },
            '/api/stats': {
                'method': 'GET',
                'description': 'Get system statistics',
                'response': {
                    'total_chunks': 'integer',
                    'total_books': 'integer',
                    'status': 'string'
                }
            }
        }
    })

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
    """
    Answer a medical question - Supports BOTH formats:
    1. Evaluation format: {"query": "...", "top_k": 5} -> {"answer": "...", "contexts": [...]}
    2. Web UI format: {"question": "...", "top_k": 3} -> {"answer": "...", "sources": [...]}
    """
    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        
        # Support both 'query' (evaluation) and 'question' (web UI)
        question = data.get('query') or data.get('question', '')
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({'error': 'Query/Question is required'}), 400
        
        # Get answer from RAG system
        result = rag.answer_question(question, top_k=top_k)
        
        # Return evaluation format if 'query' was sent
        if 'query' in data:
            # EVALUATION FORMAT: {"answer": "string", "contexts": ["string", ...]}
            response = {
                'answer': result['answer'],
                'contexts': [s['text'] for s in result['sources'][:top_k]]
            }
        else:
            # WEB UI FORMAT: {"answer": "...", "sources": [...]}
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
        
        return jsonify(response), 200
        
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
        
        # Perform semantic search
        results = rag.search(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
