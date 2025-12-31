# Quick Start Guide - Run Commands

## Prerequisites

Ensure you have Python 3.10+ installed and all dependencies installed.

---

## Installation

### Install All Dependencies

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

---

## Running the Application

### Option 1: Run Both Servers Manually

#### Terminal 1 - Start Backend (FastAPI)
```bash
cd backend
python main.py
```

**Expected Output:**
```
Loading StripUnetMCSA model...
Loading StripUnetMCSA (ResNet50 + 5-Stage Decoder)
Model path: C:\Users\91812\Desktop\stripunet_app\backend\best_f1_0.778.pt
Device: cpu
✓ Model loaded successfully!
  Total parameters: 40.4M
  Performance: F1=77.8% (World #1)
Model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Backend URL:** http://localhost:8000  
**API Docs:** http://localhost:8000/docs

---

#### Terminal 2 - Start Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Frontend URL:** http://localhost:8501

---

### Option 2: Headless Mode (No Auto Browser Open)

#### Backend (Same as above)
```bash
cd backend
python main.py
```

#### Frontend (Headless)
```bash
cd frontend
streamlit run app.py --server.headless=true
```

Opens on http://localhost:8501 but won't auto-launch browser.

---

## Stopping the Application

### Stop Backend
- Press `Ctrl+C` in the backend terminal

### Stop Frontend
- Press `Ctrl+C` in the frontend terminal

---

## Custom Configuration

### Run Backend on Different Port
```python
# Edit backend/main.py, change:
uvicorn.run(app, host="0.0.0.0", port=8000)
# To:
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Run Frontend on Different Port
```bash
streamlit run app.py --server.port=8502
```

### Change Backend URL in Frontend
1. Open http://localhost:8501
2. In sidebar, change "Backend URL" to your custom URL
3. Or edit `frontend/app.py` and change default value

---

## Testing the Setup

### 1. Test Backend Health
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "model_parameters": "40.4M",
    "performance": "F1=77.8%"
}
```

### 2. Test Frontend
- Open http://localhost:8501 in browser
- Should see "✅ Connected to backend - Model loaded"

---

## Production Deployment

### Backend (with Uvicorn workers)
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend (Production mode)
```bash
cd frontend
streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false
```

---

## Troubleshooting

### Backend doesn't start
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# Kill process if needed (Windows)
taskkill /PID <process_id> /F
```

### Frontend doesn't connect to backend
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check firewall settings
# Ensure port 8000 is not blocked
```

### Dependencies missing
```bash
# Reinstall all dependencies
cd backend
pip install -r requirements.txt --force-reinstall

cd ../frontend
pip install -r requirements.txt --force-reinstall
```

---

## Quick Reference

| Component | Command | Port | URL |
|-----------|---------|------|-----|
| **Backend** | `python main.py` | 8000 | http://localhost:8000 |
| **Frontend** | `streamlit run app.py` | 8501 | http://localhost:8501 |
| **API Docs** | - | 8000 | http://localhost:8000/docs |

---

## Environment Setup (First Time)

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt

cd ../frontend
pip install -r requirements.txt
```

### 3. Verify Model File
```bash
# Ensure model file exists
dir backend\best_f1_0.778.pt

# Should show a file ~161 MB in size
```

### 4. Run Application
Follow "Running the Application" section above.

---

## Development Mode

### Auto-reload Backend (using --reload)
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Changes to Python files will auto-reload the server.

### Auto-reload Frontend
Streamlit automatically reloads on file changes - no special flag needed.

---

## Logs and Monitoring

### View Backend Logs
Logs are printed to terminal where `python main.py` is running.

### View Frontend Logs
Logs are printed to terminal where `streamlit run app.py` is running.

### Access API Request Logs
Backend logs all requests:
```
INFO:     127.0.0.1:54815 - "POST /predict HTTP/1.1" 200 OK
```

---

## System Requirements

- **Python:** 3.10 or higher
- **RAM:** Minimum 2GB (4GB recommended)
- **Disk:** ~500MB for dependencies + model
- **OS:** Windows, Linux, or macOS
- **Network:** Ports 8000 and 8501 available

---

## Ready to Use!

1. Open **Terminal 1** → Run backend
2. Open **Terminal 2** → Run frontend
3. Open browser → http://localhost:8501
4. Upload satellite image
5. Click "🚀 Run Segmentation"
6. Get results! 🎉

---

**Need Help?** Check troubleshooting section or review the detailed documentation:
- [BACKEND_SUMMARY.md](BACKEND_SUMMARY.md) - Complete backend documentation
- [FRONTEND_SUMMARY.md](FRONTEND_SUMMARY.md) - Complete frontend documentation
- [MODEL_SUMMARY.md](MODEL_SUMMARY.md) - Model architecture details
