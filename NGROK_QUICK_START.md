# 🚀 Quick Ngrok Deployment

## ⚡ 3-Step Deployment

### 1. Start the Server
```powershell
cd "d:\hack acure\Dataset"
python app.py
```

### 2. Install & Setup Ngrok
```powershell
# Install ngrok
winget install ngrok.ngrok

# Sign up at: https://dashboard.ngrok.com/signup
# Get token from: https://dashboard.ngrok.com/get-started/your-authtoken

# Configure (one-time)
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 3. Start Ngrok Tunnel
```powershell
ngrok http 5000
```

Copy the `https://` URL and share it! 🎉

---

## 📱 What You Get

A beautiful web interface where users can:
- ✅ Ask medical questions
- ✅ Get AI-powered answers
- ✅ See source citations
- ✅ Browse multiple medical books

---

## 🔗 Example

After running ngrok, you'll see:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:5000
```

Share: `https://abc123.ngrok-free.app`

Anyone can access your Medical RAG system! 🌍

---

## 📖 Full Guide

See `DEPLOY_NGROK.md` for:
- API endpoints
- Security settings
- Custom domains
- Troubleshooting
