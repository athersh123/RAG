# Medical RAG - Quick Start Script for Ngrok Deployment

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Medical RAG System - Ngrok Deployment" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Navigate to project
Set-Location "d:\hack acure"

Write-Host "Step 1: Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found! Install Python 3.12+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK`n" -ForegroundColor Green

Write-Host "Step 2: Checking dependencies..." -ForegroundColor Yellow
$packages = @("flask", "flask-cors", "sentence-transformers", "torch", "numpy")
foreach ($pkg in $packages) {
    python -c "import $($pkg.replace('-','_'))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $pkg missing - installing..." -ForegroundColor Yellow
        pip install $pkg --quiet
    }
}

Write-Host "`nStep 3: Checking embeddings..." -ForegroundColor Yellow
$embFiles = Get-ChildItem "Embeddings\*.npy" -ErrorAction SilentlyContinue
if ($embFiles.Count -gt 0) {
    Write-Host "  ✓ Found $($embFiles.Count) embedding files" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  No embeddings found!" -ForegroundColor Yellow
    Write-Host "  Creating sample data..." -ForegroundColor Yellow
    Set-Location "Dataset"
    python create_sample_data.py
    Set-Location ".."
}

Write-Host "`nStep 4: Starting Flask server..." -ForegroundColor Yellow
Start-Process python -ArgumentList "Dataset\app.py" -WindowStyle Normal
Start-Sleep -Seconds 3
Write-Host "  ✓ Server started on http://localhost:5000" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "1. Test locally first:" -ForegroundColor Yellow
Write-Host "   http://localhost:5000`n" -ForegroundColor White

Write-Host "2. Install ngrok (if not installed):" -ForegroundColor Yellow
Write-Host "   winget install ngrok.ngrok`n" -ForegroundColor White

Write-Host "3. Sign up at ngrok.com (free):" -ForegroundColor Yellow
Write-Host "   https://dashboard.ngrok.com/signup`n" -ForegroundColor White

Write-Host "4. Get your auth token:" -ForegroundColor Yellow
Write-Host "   https://dashboard.ngrok.com/get-started/your-authtoken`n" -ForegroundColor White

Write-Host "5. Configure ngrok:" -ForegroundColor Yellow
Write-Host "   ngrok config add-authtoken YOUR_TOKEN`n" -ForegroundColor White

Write-Host "6. Start ngrok tunnel:" -ForegroundColor Yellow
Write-Host "   ngrok http 5000`n" -ForegroundColor White

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quick Commands:" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "# Open local web interface" -ForegroundColor Green
Write-Host "start http://localhost:5000`n" -ForegroundColor White

Write-Host "# Start ngrok (in new terminal)" -ForegroundColor Green
Write-Host "ngrok http 5000`n" -ForegroundColor White

Write-Host "`n📖 Full guide: DEPLOY_NGROK.md`n" -ForegroundColor Cyan

# Open browser
Start-Sleep -Seconds 2
Write-Host "Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:5000"
