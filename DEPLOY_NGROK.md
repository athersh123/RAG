# 🌐 Deploy Medical RAG on Ngrok

Complete guide to deploy your Medical RAG system online using ngrok.

---

## 📋 Prerequisites

1. **Python 3.12+** installed
2. **Medical RAG repository** cloned
3. **Embeddings generated** (or sample data created)

---

## 🚀 Quick Deployment (5 minutes)

### Step 1: Install Dependencies

```powershell
cd "d:\hack acure"
pip install flask flask-cors
```

### Step 2: Install ngrok

**Option A: Using winget (Recommended)**
```powershell
winget install --id=ngrok.ngrok -e
```

**Option B: Manual Download**
1. Go to: https://ngrok.com/download
2. Download for Windows
3. Extract to a folder
4. Add to PATH or use full path

### Step 3: Sign up for ngrok (Free)

1. Visit: https://dashboard.ngrok.com/signup
2. Sign up (free account)
3. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken

### Step 4: Configure ngrok

```powershell
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

### Step 5: Start the Flask Server

```powershell
cd "d:\hack acure\Dataset"
python app.py
```

You should see:
```
======================================================================
🌐 Medical RAG Web Server
======================================================================

📍 Local URL: http://localhost:5000
📍 Network URL: http://0.0.0.0:5000
```

### Step 6: Start ngrok (in a NEW terminal)

```powershell
ngrok http 5000
```

You'll see output like:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:5000
```

### Step 7: Access Your App

Copy the `https://` URL from ngrok and share it!

**Example:** `https://abc123.ngrok-free.app`

Anyone can access your Medical RAG system via this URL! 🎉

---

## 📱 Using the Web Interface

Once deployed, users can:

1. **Visit the URL** - Opens beautiful web interface
2. **Ask questions** - Type medical questions
3. **Get answers** - View AI-powered responses with sources
4. **See citations** - Check which medical textbook answered

### Example Questions:
- "What is diabetic ketoacidosis?"
- "How is hypertension treated?"
- "What are symptoms of heart failure?"

---

## 🔧 Advanced Configuration

### Custom Domain (ngrok Pro)

```powershell
ngrok http 5000 --domain=your-custom-domain.ngrok-free.app
```

### Password Protection

Add basic auth to ngrok:
```powershell
ngrok http 5000 --basic-auth="username:password"
```

### HTTPS Only

```powershell
ngrok http 5000 --scheme=https
```

---

## 📊 API Endpoints

Your deployed app has these endpoints:

### Web Interface
- `GET /` - Beautiful web UI

### API Endpoints
- `POST /api/ask` - Ask a question
- `POST /api/search` - Search medical knowledge
- `GET /api/stats` - Get system statistics
- `GET /health` - Health check

### Example API Usage:

```python
import requests

url = "https://your-ngrok-url.ngrok-free.app/api/ask"
response = requests.post(url, json={
    "question": "What is diabetes?",
    "top_k": 3
})
print(response.json())
```

```javascript
// JavaScript
fetch('https://your-ngrok-url.ngrok-free.app/api/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        question: "What is diabetes?",
        top_k: 3
    })
})
.then(r => r.json())
.then(data => console.log(data));
```

---

## 🛡️ Security Considerations

### For Public Deployment:

1. **Add Rate Limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["10 per minute"])
```

2. **Add API Key Authentication**
```python
@app.before_request
def check_api_key():
    api_key = request.headers.get('X-API-Key')
    if api_key != 'your-secret-key':
        return jsonify({'error': 'Unauthorized'}), 401
```

3. **Enable CORS Selectively**
```python
CORS(app, origins=["https://yourdomain.com"])
```

4. **Use Environment Variables**
```python
import os
SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret')
```

---

## 🔄 Keep Server Running

### Option 1: Keep Terminal Open
Just leave the terminal running with `python app.py`

### Option 2: Run in Background (Windows)
```powershell
Start-Process python -ArgumentList "app.py" -WindowStyle Hidden
```

### Option 3: Use Screen/Tmux (Linux/Mac)
```bash
screen -S rag
python app.py
# Press Ctrl+A, then D to detach
```

---

## 📈 Monitor Your Deployment

### ngrok Dashboard
Visit: https://dashboard.ngrok.com/observability/traffic-inspection

See:
- Request logs
- Traffic analytics
- Response times
- Error rates

### Flask Logs
Watch the terminal running `app.py` for:
- Incoming requests
- Errors
- Performance metrics

---

## 🐛 Troubleshooting

### ngrok not found
```powershell
# Use full path
C:\Users\YourName\ngrok\ngrok.exe http 5000
```

### Port already in use
```powershell
# Change port in app.py
app.run(host='0.0.0.0', port=5001)

# Then use:
ngrok http 5001
```

### Embeddings not found
```powershell
# Generate sample data
cd Dataset
python create_sample_data.py
```

### Slow responses
- Reduce `top_k` in API calls
- Use smaller embedding models
- Add caching

---

## 💰 Ngrok Pricing

| Plan | Cost | Features |
|------|------|----------|
| **Free** | $0 | 1 online tunnel, random URLs, 40 requests/min |
| **Personal** | $8/mo | 3 tunnels, custom domains, 60 req/min |
| **Pro** | $20/mo | 10 tunnels, reserved domains, no limits |

**Free plan is fine for testing and small projects!**

---

## 🌟 Alternative Deployment Options

### 1. Heroku (Free Tier)
```powershell
# Install Heroku CLI
heroku create medical-rag
git push heroku main
```

### 2. Render (Free)
1. Push to GitHub
2. Connect to render.com
3. Deploy automatically

### 3. Railway (Free $5 credit)
1. Push to GitHub
2. Connect to railway.app
3. One-click deploy

### 4. Local Network Only
```powershell
# Just run app.py
python app.py
# Access via: http://YOUR_LOCAL_IP:5000
```

---

## 📝 Complete Deployment Script

Save as `deploy.ps1`:

```powershell
# Medical RAG Deployment Script

Write-Host "🚀 Starting Medical RAG Deployment..." -ForegroundColor Green

# Check Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
pip install flask flask-cors sentence-transformers torch numpy

# Start Flask server
Write-Host "`n🌐 Starting Flask server..." -ForegroundColor Yellow
Start-Process python -ArgumentList "Dataset/app.py" -PassThru

Start-Sleep -Seconds 3

# Start ngrok
Write-Host "`n🔗 Starting ngrok..." -ForegroundColor Yellow
Write-Host "   Copy the HTTPS URL and share it!" -ForegroundColor Cyan
ngrok http 5000
```

Run with:
```powershell
powershell -ExecutionPolicy Bypass -File deploy.ps1
```

---

## ✅ Success Checklist

- [ ] Flask installed (`pip install flask flask-cors`)
- [ ] ngrok installed and authenticated
- [ ] Embeddings generated (or sample data created)
- [ ] Flask server running (`python app.py`)
- [ ] ngrok tunnel active (`ngrok http 5000`)
- [ ] Public URL shared with users
- [ ] Web interface accessible

---

## 🎉 You're Live!

Share your ngrok URL with anyone:
- **Web Interface**: `https://your-url.ngrok-free.app`
- **API Endpoint**: `https://your-url.ngrok-free.app/api/ask`

Your Medical RAG system is now accessible worldwide! 🌍

---

## 📞 Support

Issues? Check:
1. Flask server is running (check terminal)
2. ngrok is running (check second terminal)
3. Embeddings exist (check `Embeddings/` folder)
4. Firewall allows port 5000

Need help? Open an issue on GitHub!
