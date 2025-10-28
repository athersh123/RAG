# Medical RAG API Guide

## 🌐 Public Endpoint
**Base URL:** `https://unprocrastinated-cheryll-beneficent.ngrok-free.dev`

---

## 📋 API Endpoints

### 1. **POST /api/ask** (Evaluation Endpoint)
**Purpose:** Ask a medical question and get an answer with context

**Request:**
```bash
curl -X POST https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the pregnancy category for acyclovir?",
    "top_k": 3
  }'
```

**Request Body:**
```json
{
  "query": "string (required) - Your medical question",
  "top_k": "integer (optional, default: 3) - Number of context snippets to return"
}
```

**Response (200 OK):**
```json
{
  "answer": "Acyclovir is concentrated in milk, in which levels may be higher than in plasma...",
  "contexts": [
    "Acyclovir is concentrated in milk, in which levels...",
    "Fomepizole is a pregnancy category C drug...",
    "Intravenous acyclovir, 10 mg/kg every 8 hours..."
  ]
}
```

**Response Format:**
- `answer`: String - Concise answer to the question
- `contexts`: Array of strings - Relevant text snippets from medical textbooks

---

### 2. **GET /api/ask**
**Purpose:** Get information about the endpoint (if accessed with GET instead of POST)

**Response (405 Method Not Allowed):**
```json
{
  "error": "Method Not Allowed",
  "message": "This endpoint only accepts POST requests",
  "usage": {
    "method": "POST",
    "url": "/api/ask",
    "headers": {"Content-Type": "application/json"},
    "body": {
      "query": "Your medical question here",
      "top_k": 3
    }
  },
  "expected_response": {
    "answer": "string",
    "contexts": ["string", "..."]
  }
}
```

---

### 3. **GET /health**
**Purpose:** Check if the API is running

**Request:**
```bash
curl https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Medical RAG System is running"
}
```

---

### 4. **GET /api** or **GET /api/**
**Purpose:** Get complete API documentation

**Request:**
```bash
curl https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api
```

**Response:**
```json
{
  "name": "Medical RAG API",
  "version": "1.0",
  "description": "AI-Powered Medical Knowledge Assistant",
  "endpoints": {
    "/health": { ... },
    "/api/ask": { ... },
    "/api/search": { ... },
    "/api/stats": { ... }
  }
}
```

---

### 5. **GET /api/stats**
**Purpose:** Get system statistics

**Request:**
```bash
curl https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api/stats
```

**Response:**
```json
{
  "total_chunks": 6609,
  "total_books": 10,
  "status": "ready"
}
```

---

### 6. **POST /api/search**
**Purpose:** Semantic search in medical knowledge base

**Request:**
```bash
curl -X POST https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes treatment",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "query": "diabetes treatment",
  "results": [
    {
      "book": "InternalMedicine",
      "similarity": 0.85,
      "text": "...",
      "paragraph_id": 123,
      "chunk_index": 45
    }
  ]
}
```

---

### 7. **GET /** (Web Interface)
**Purpose:** Access the interactive web UI

**Usage:** Open in browser
```
https://unprocrastinated-cheryll-beneficent.ngrok-free.dev
```

---

## 🎯 For Evaluation/Competition

### Submission URL:
```
https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api/ask
```

### Key Points:
- ✅ Method: **POST**
- ✅ Request: `{"query": "string", "top_k": integer}`
- ✅ Response: `{"answer": "string", "contexts": ["string", ...]}`
- ✅ Status: **200 OK** on success
- ✅ Timeout: Responds within 60 seconds
- ✅ Contexts: Plain strings (not objects)

---

## 🧪 Testing

### Local Test:
```bash
cd "d:\hack acure"
python test_api.py
```

### Remote Test:
```bash
curl -X POST https://unprocrastinated-cheryll-beneficent.ngrok-free.dev/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"When to give Tdap booster?","top_k":3}'
```

---

## 📊 System Information

- **Medical Textbooks:** 10 books
- **Total Knowledge Chunks:** 6,609
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Domains Covered:**
  - Anatomy & Physiology
  - Cardiology
  - Dentistry
  - Emergency Medicine
  - Gastroenterology
  - General Medicine
  - Infectious Disease
  - Internal Medicine
  - Nephrology

---

## 🔧 Troubleshooting

### "Method Not Allowed" Error
- **Cause:** Using GET instead of POST
- **Solution:** Use POST method with the `/api/ask` endpoint

### "Connection Refused" Error
- **Cause:** Flask server not running
- **Solution:** Run `.\deploy.bat` to start both Flask and ngrok

### "Endpoint Offline" Error
- **Cause:** Ngrok tunnel is active but Flask server crashed
- **Solution:** Restart Flask server: `cd Dataset ; python app.py`

---

## 📝 Notes

- The Flask server must be running for the API to work
- Ngrok provides the public URL but needs Flask backend
- Both servers are started automatically with `deploy.bat`
- Check minimized windows for Flask and Ngrok processes
- Ngrok dashboard: http://127.0.0.1:4040

---

## 🚀 Quick Start

```powershell
# Start deployment
cd "d:\hack acure"
.\deploy.bat

# Get ngrok URL
# Open http://127.0.0.1:4040 in browser

# Test API
python test_api.py
```
