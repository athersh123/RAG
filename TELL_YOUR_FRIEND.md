# 🚀 Quick Start for Your Friend

**Share this with your friend who got the "No embedding files found" error!**

---

## The Problem
The repository doesn't include the large embedding files (~200 MB) because they were excluded from Git.

## The Solution (Choose ONE)

### ⚡ FASTEST: Use Sample Data (2 minutes)

```powershell
# 1. Clone the repo
git clone https://github.com/athersh123/RAG.git
cd RAG

# 2. Install dependencies
pip install sentence-transformers torch numpy

# 3. Create sample embeddings
cd Dataset
python create_sample_data.py

# 4. Test the system
python train.py question "What is diabetes?"
```

**Done!** The system will work with sample medical data.

---

### 🔧 FULL SETUP: With Medical PDFs (20 minutes)

If your friend has medical textbook PDFs:

```powershell
# 1-2. Same as above

# 3. Add PDFs to Dataset/ folder
# Place your medical PDFs in:
#   RAG/Dataset/Cardiology.pdf
#   RAG/Dataset/General.pdf
#   etc.

# 4. Generate embeddings
cd Dataset
python generate_embeddings.py

# 5. Run the system
python train.py
```

---

## 📋 What Each Method Gives You

| Method | Setup Time | Data Size | Functionality |
|--------|-----------|-----------|---------------|
| **Sample Data** | 2 min | ~100 KB | Basic testing, 9 medical facts |
| **Full PDFs** | 20 min | ~200 MB | Full system, 43,258 medical chunks |

---

## 🎯 Tell Your Friend:

**"The repository works, but you need to generate the embeddings first!"**

**Quick test (2 minutes):**
```
pip install sentence-transformers torch numpy
cd Dataset
python create_sample_data.py
python train.py
```

**Or read the full guide:** `SETUP_FOR_NEW_USERS.md`

---

## 📞 If They Still Have Issues

1. Make sure Python 3.12+ is installed
2. Make sure pip install worked
3. Check they're in the right directory
4. See `SETUP_FOR_NEW_USERS.md` for troubleshooting

---

## ✅ Expected Output After Setup

```
######################################################################
# Medical Knowledge RAG System
# Retrieval-Augmented Generation for Medical Questions
######################################################################

📚 Loading medical knowledge base...
   ✓ General: 3 chunks
   ✓ Cardiology: 2 chunks
   ✓ Emergency: 2 chunks
   ✓ Gastrology: 2 chunks

✅ Loaded 4 medical textbooks
📊 Total knowledge chunks: 9
```

If they see this, **it's working!** ✅
