# 🚀 Setup Guide for New Users

**If you cloned this repository and got an error about "No embedding files found"**, follow this guide!

## ⚠️ Why This Error Happens

The repository **excludes large files** (~2 GB) from Git:
- Medical textbook PDFs (~1.6 GB)
- Embedding files (~200 MB)
- Cleaned CSV data (~214 MB)

**You need to generate these files yourself!**

---

## 📋 Two Options to Run This Project

### Option 1: **With Medical PDFs** (Full Setup - Recommended)
If you have medical textbook PDFs, you can generate embeddings.

### Option 2: **Demo/Test Mode** (Quick Setup)
Use sample data to test the system without full embeddings.

---

## 🔧 Option 1: Full Setup (With PDFs)

### Step 1: Install Dependencies
```powershell
cd path\to\RAG
pip install -r requirements.txt
```

### Step 2: Add Your Medical PDFs
Place your medical textbook PDFs in the `Dataset/` folder:
```
Dataset/
├── Anatomy&Physiology.pdf
├── Cardiology.pdf
├── Dentistry.pdf
├── EmergencyMedicine.pdf
├── Gastrology.pdf
├── General.pdf
├── InfectiousDisease.pdf
├── InternalMedicine.pdf
└── Nephrology.pdf
```

**Note:** You need to obtain these PDFs legally (library, purchase, academic access, etc.)

### Step 3: Generate Embeddings
```powershell
cd Dataset
python generate_embeddings.py
```

This will:
- Extract text from PDFs
- Clean the text
- Generate embeddings (~15-20 minutes)
- Create files in `Embeddings/` folder

### Step 4: Run the RAG System
```powershell
python train.py
```

---

## ⚡ Option 2: Quick Test Mode (Without PDFs)

If you don't have medical PDFs but want to test the code:

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Create Sample Embeddings

Create a file `Dataset/create_sample_data.py`:

```python
import numpy as np
import json
from pathlib import Path

# Create Embeddings directory
embeddings_dir = Path(__file__).parent.parent / 'Embeddings'
embeddings_dir.mkdir(exist_ok=True)

# Sample medical text
sample_texts = [
    "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels.",
    "Hypertension is defined as blood pressure consistently above 140/90 mmHg.",
    "Heart failure occurs when the heart cannot pump blood effectively.",
    "Pneumonia is an infection of the lungs caused by bacteria, viruses, or fungi.",
    "Gastritis is inflammation of the stomach lining."
]

# Create sample embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(sample_texts)
metadata = [{"text": text, "source_file": "Sample", "chunk_id": i} 
            for i, text in enumerate(sample_texts)]

# Save sample data
np.save(embeddings_dir / 'Sample_embeddings.npy', embeddings)
with open(embeddings_dir / 'Sample_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("✓ Sample embeddings created!")
```

### Step 3: Run Sample Data Generator
```powershell
cd Dataset
python create_sample_data.py
```

### Step 4: Test the System
```powershell
python train.py question "What is diabetes?"
```

---

## 🌐 Alternative: Download Pre-generated Embeddings

**If the repository owner provides embeddings separately:**

### Option A: Google Drive/Dropbox Link
1. Download the `Embeddings.zip` file from the shared link
2. Extract to your project folder
3. Run `python Dataset/train.py`

### Option B: Git LFS (Large File Storage)
If embeddings are hosted via Git LFS:
```powershell
git lfs install
git lfs pull
```

---

## 📝 What Each File Does

| File | Purpose | Needed? |
|------|---------|---------|
| `train.py` | Main RAG system | ✅ Yes (in Git) |
| `generate_embeddings.py` | Creates embeddings from PDFs | ✅ Yes (in Git) |
| `Dataset/*.pdf` | Medical textbooks | ❌ Not in Git |
| `Embeddings/*.npy` | Vector embeddings | ❌ Not in Git |
| `Cleaned_Data/*.csv` | Cleaned text | ❌ Not in Git |

---

## 🔍 Troubleshooting

### Error: "No embedding files found"
**Solution:** You need to either:
1. Generate embeddings (Option 1 above)
2. Create sample data (Option 2 above)
3. Download pre-generated embeddings from repository owner

### Error: "No module named 'sentence_transformers'"
**Solution:**
```powershell
pip install sentence-transformers torch
```

### Error: "PDF file not found"
**Solution:** Place medical PDF files in `Dataset/` folder

### Embeddings take too long
**Solution:** 
- Start with 1-2 PDFs first
- Uses ~200 MB RAM per PDF
- Takes ~2-3 minutes per PDF

---

## 📊 Expected Directory Structure After Setup

```
RAG/
├── Dataset/
│   ├── train.py                    ✅ (from Git)
│   ├── generate_embeddings.py      ✅ (from Git)
│   ├── Cardiology.pdf              ❌ (you add)
│   └── ...other PDFs               ❌ (you add)
├── Embeddings/
│   ├── Cardiology_embeddings.npy   ❌ (generated)
│   ├── Cardiology_metadata.json    ❌ (generated)
│   └── ...other embeddings         ❌ (generated)
├── Cleaned_Data/                   ❌ (auto-generated)
└── requirements.txt                ✅ (from Git)
```

---

## 💡 Quick Start Summary

**Minimal setup to test the code:**
```powershell
# 1. Clone repo
git clone https://github.com/athersh123/RAG.git
cd RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create sample data (for testing)
cd Dataset
python create_sample_data.py

# 4. Run the system
python train.py
```

**Full setup with your own PDFs:**
```powershell
# 1-2. Same as above

# 3. Add PDFs to Dataset/

# 4. Generate embeddings
cd Dataset
python generate_embeddings.py

# 5. Run the system
python train.py
```

---

## 📞 Need Help?

- Check `HOW_TO_RUN.md` for usage instructions
- Check `README.md` for project overview
- Open an issue on GitHub
- Contact repository owner

---

## ⚖️ Legal Note

**Medical textbooks are copyrighted material.** You must:
- ✅ Own the physical/digital books
- ✅ Have institutional access (university library)
- ✅ Use for personal/educational purposes only
- ❌ DO NOT share copyrighted PDFs publicly

This project is for educational purposes only.
