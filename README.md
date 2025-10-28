# 🏥 Medical RAG System

A **Retrieval-Augmented Generation (RAG)** system for medical knowledge base querying, built with Python, sentence-transformers, and 9 comprehensive medical textbooks.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This RAG system enables intelligent medical question-answering by:
1. **Indexing** 9 medical textbooks (~18,000 pages) across multiple specialties
2. **Embedding** medical knowledge into 43,258 semantic vectors (384 dimensions)
3. **Retrieving** relevant context using cosine similarity search
4. **Generating** accurate answers with source citations

### Medical Domains Covered
- 🫀 **Cardiology** - Heart and cardiovascular diseases
- 🦷 **Dentistry** - Oral and dental health
- 🚑 **Emergency Medicine** - Acute care and emergencies
- 🫁 **Gastrology** - Digestive system disorders
- 🩺 **General Medicine** - Primary care and common conditions
- 🦠 **Infectious Disease** - Bacterial, viral, and parasitic infections
- 🏥 **Internal Medicine** - Adult disease management
- 🧬 **Nephrology** - Kidney and renal diseases
- 🧠 **Anatomy & Physiology** - Human body structure and function

## ✨ Features

- 🔍 **Semantic Search**: Find relevant medical information across 43,258 knowledge chunks
- 💬 **Interactive Chat**: Ask questions in natural language
- 📚 **Multi-Source Retrieval**: Answers cite specific medical textbooks
- ⚡ **Fast Inference**: Sub-second search with pre-computed embeddings
- 🤖 **LLM Support**: Optional integration with OpenAI, Ollama, or GPT4All
- 📊 **Quality Metrics**: 90.3/100 data quality, 100/100 embedding accuracy

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.12+
pip install sentence-transformers torch numpy pandas scikit-learn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/athersh123/RAG.git
cd RAG
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate embeddings** (if not included)
```bash
# Place your PDF files in Dataset/ folder
cd Dataset
python generate_embeddings.py
```

4. **Run the RAG system**
```bash
python train.py
```

## 💻 Usage

### Interactive Mode
```bash
cd Dataset
python train.py
```
Then type your medical questions:
```
❓ Your question: What is diabetic ketoacidosis?
🔍 Searching medical knowledge base...
   ✓ Found 3 relevant sources
💡 Answer: [Detailed medical information...]
```

### Single Question
```bash
python train.py question "What are the symptoms of heart failure?"
```

### Search Mode
```bash
python train.py search "hypertension treatment"
```

### Demo Mode
```bash
python train.py demo
```

## 📊 System Architecture

```
User Query
    ↓
Query Embedding (384-dim vector)
    ↓
Cosine Similarity Search
    ↓
Top-K Medical Contexts Retrieved
    ↓
LLM Generation (Optional)
    ↓
Answer + Citations
```

### Technology Stack
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector Store**: NumPy arrays (43,258 × 384)
- **Search**: Cosine similarity with scikit-learn
- **LLM Options**: OpenAI GPT, Ollama, GPT4All (optional)

## 📁 Project Structure

```
RAG/
├── Dataset/
│   ├── train.py                    # Main RAG system
│   ├── generate_embeddings.py      # Embedding generation
│   ├── clean_data.py               # Text preprocessing
│   ├── validate_embeddings.py      # Quality checks
│   └── [Medical PDFs - not tracked in git]
├── Embeddings/                     # Pre-computed embeddings (not in git)
│   ├── *_embeddings.npy           # Vector arrays (63 MB)
│   ├── *_metadata.json            # Source text metadata
│   └── *_vectors.pkl              # Combined format
├── Cleaned_Data/                   # Preprocessed text (not in git)
│   └── *_cleaned_fixed.csv        # Cleaned paragraphs
├── HOW_TO_RUN.md                  # Detailed usage guide
├── README_RAG.md                  # RAG implementation notes
└── requirements.txt               # Python dependencies
```

## 🔧 Configuration

### Using OpenAI for Better Answers
```bash
export OPENAI_API_KEY="your-api-key"
python train.py --model-type openai --model-name gpt-3.5-turbo question "your question"
```

### Using Ollama (Free Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:1b

# Use with RAG
python train.py --model-type ollama --model-name llama3.2:1b question "your question"
```

### Command Line Arguments
- `--model-type`: `demo`, `openai`, `ollama`, `gpt4all` (default: `demo`)
- `--model-name`: Model identifier (e.g., `gpt-3.5-turbo`, `llama3.2:1b`)
- `--top-k`: Number of sources to retrieve (default: `3`)
- `--temperature`: Answer creativity 0-1 (default: `0.7`)

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Embeddings** | 43,258 chunks |
| **Vector Dimensions** | 384 |
| **Data Quality Score** | 90.3/100 |
| **Embedding Accuracy** | 100/100 |
| **Search Speed** | <1 second |
| **Memory Usage** | ~200 MB |

### Embedding Distribution by Domain
- Gastrology: 33,418 chunks (77%)
- Nephrology: 4,845 chunks (11%)
- General Medicine: 3,083 chunks (7%)
- Emergency Medicine: 1,267 chunks (3%)
- Others: 645 chunks (2%)

## 🎯 Example Queries

### Good Questions (High Accuracy)
✅ "What is diabetic ketoacidosis?"
✅ "How is pneumonia diagnosed?"
✅ "What are the symptoms of heart failure?"
✅ "Explain the pathophysiology of hypertension"
✅ "What is the treatment for acute pancreatitis?"

### Limited Questions (May Need External Knowledge)
⚠️ "What is the FDA pregnancy category for drug X?"
⚠️ "Latest 2025 treatment guidelines"
⚠️ "Compare medication A vs B efficacy studies"

## 🔬 Data Pipeline

1. **PDF Extraction** → PyPDF2
2. **Text Cleaning** → Remove duplicates, fix encoding, normalize text
3. **Validation** → Quality scoring (90.3/100)
4. **Chunking** → 512 words per chunk, 50-word overlap
5. **Embedding** → all-MiniLM-L6-v2 (384-dim vectors)
6. **Validation** → Semantic similarity tests (100/100)
7. **Indexing** → NumPy arrays + metadata JSON

## ⚠️ Limitations

- **Dataset Coverage**: Limited to 9 medical textbooks (may not cover all specialties)
- **No Real-time Updates**: Knowledge cutoff at textbook publication dates
- **No Clinical Advice**: For educational purposes only, not medical diagnosis
- **English Only**: Current implementation supports English medical texts

## 🛠️ Development

### Generate New Embeddings
```bash
cd Dataset
python generate_embeddings.py
```

### Validate Data Quality
```bash
python validate_cleaned_data.py
python validate_embeddings.py
```

### Add New Medical Textbooks
1. Place PDF in `Dataset/` folder
2. Run `python generate_embeddings.py`
3. New embeddings will be added to `Embeddings/` folder

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **sentence-transformers**: Embedding generation
- **PyTorch**: Deep learning framework
- **scikit-learn**: Similarity search
- Medical textbook authors and publishers

## 📧 Contact

**Author**: Athersh  
**GitHub**: [@athersh123](https://github.com/athersh123)  
**Repository**: [RAG](https://github.com/athersh123/RAG)

## ⚖️ Disclaimer

This RAG system is for **educational and research purposes only**. It is not intended for clinical diagnosis, treatment decisions, or as a substitute for professional medical advice. Always consult qualified healthcare professionals for medical concerns.

---

**⭐ Star this repo if you find it useful!**
