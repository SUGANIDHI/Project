# StripUnetMCSA Deployment Guide for Render

## 🚀 Deploying to Render

This guide will help you deploy the StripUnetMCSA application to Render.com with both backend and frontend services.

---

## Prerequisites

1. ✅ GitHub repository: https://github.com/SUGANIDHI/Project
2. ✅ Render account: https://render.com
3. ⚠️ Model weights file (`best_f1_0.778.pt` - 161MB)

---

## Important: Model Weights

The model file is **161MB** and cannot be stored directly in Git. You have two options:

### Option 1: Use Cloud Storage (Recommended)
Upload `best_f1_0.778.pt` to cloud storage (Google Drive, Dropbox, AWS S3) and modify `backend/model_loader.py` to download it on startup.

### Option 2: Manual Upload via Render Dashboard
After deployment, use Render Shell to upload the model file directly.

---

## Deployment Steps

### Step 1: Prepare Repository

The repository already has these deployment files:
- ✅ `render.yaml` - Main deployment configuration
- ✅ `backend/build.sh` - Backend build script
- ✅ `backend/start.sh` - Backend start script
- ✅ `frontend/start.sh` - Frontend start script

### Step 2: Create Backend Service

1. Go to Render Dashboard: https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `SUGANIDHI/Project`
4. Configure the backend:
   - **Name**: `stripunet-backend`
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `bash start.sh` or `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. Add Environment Variables:
   - `PYTHON_VERSION` = `3.10.0`
   - `PORT` = (will be auto-set by Render)

6. Click **"Create Web Service"**

### Step 3: Upload Model Weights to Backend

**After backend deployment:**

1. Go to your backend service dashboard
2. Click on **"Shell"** tab
3. Upload the model file:
   ```bash
   # Option A: Download from URL (if you uploaded to cloud)
   curl -L "YOUR_MODEL_DOWNLOAD_URL" -o best_f1_0.778.pt
   
   # Option B: Use Render's persistent disk (paid feature)
   # Mount a disk and store the model there
   ```

### Step 4: Create Frontend Service

1. Click **"New +"** → **"Web Service"**
2. Connect the same repository
3. Configure the frontend:
   - **Name**: `stripunet-frontend`
   - **Region**: Same as backend
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `bash start.sh` or full streamlit command
   - **Plan**: Free

4. Add Environment Variables:
   - `PYTHON_VERSION` = `3.10.0`
   - `BACKEND_URL` = `https://stripunet-backend.onrender.com` (your backend URL)
   - `PORT` = (auto-set by Render)

5. Click **"Create Web Service"**

### Step 5: Update Frontend to Use Backend URL

Modify `frontend/app.py` to use the environment variable:

```python
import os

# Get backend URL from environment or use default
backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')

# Use this URL for all API calls
```

---

## Alternative: Using render.yaml (Blueprint)

1. Go to Render Dashboard
2. Click **"New +"** → **"Blueprint"**
3. Select your repository
4. Render will automatically detect `render.yaml` and create both services

---

## Post-Deployment Configuration

### 1. Update Frontend Backend URL

In the Streamlit app sidebar, update the default backend URL to your deployed backend:
```
https://stripunet-backend.onrender.com
```

### 2. Test the Deployment

1. Visit your frontend URL: `https://stripunet-frontend.onrender.com`
2. Upload a test image
3. Run segmentation
4. Verify results

---

## Free Tier Limitations

⚠️ **Render Free Tier Constraints:**
- Goes to sleep after 15 minutes of inactivity
- 512 MB RAM limit
- First request after sleep takes ~1 minute to wake up
- 750 hours/month free (per service)

**Model Size Issue:**
The 161MB model + dependencies may exceed free tier RAM. Consider:
- Using the paid "Starter" plan ($7/month)
- Optimizing the model (quantization, pruning)
- Using a smaller model variant

---

## Configuration Files Reference

### render.yaml
Defines both services in a single file for blueprint deployment.

### backend/start.sh
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

### frontend/start.sh
```bash
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true
```

---

## Environment Variables

### Backend
| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.10.0` | Yes |
| `PORT` | Auto-set | Yes |
| `MODEL_PATH` | `best_f1_0.778.pt` | Yes |

### Frontend
| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.10.0` | Yes |
| `PORT` | Auto-set | Yes |
| `BACKEND_URL` | Backend service URL | Yes |

---

## Troubleshooting

### Backend fails to start
- Check logs: "Logs" tab in service dashboard
- Verify model file exists: `ls -lh best_f1_0.778.pt`
- Check RAM usage: May need to upgrade to paid plan

### Frontend can't connect to backend
- Verify backend URL is correct
- Check CORS settings in `backend/main.py`
- Ensure both services are running

### Out of Memory errors
- Model (161MB) + PyTorch is heavy for free tier
- Upgrade to Starter plan ($7/month) with more RAM
- Or reduce model size/complexity

### Slow cold starts
- Free tier services sleep after 15 minutes
- First request takes ~60 seconds to wake up
- Upgrade to paid plan for always-on services

---

## Recommended: Use Paid Tier

For production use, consider Render's **Starter Plan** ($7/month):
- ✅ 512 MB RAM → 2 GB RAM
- ✅ Always on (no sleep)
- ✅ Faster cold starts
- ✅ Better performance

---

## Model Storage Solutions

### Option 1: Google Drive
1. Upload model to Google Drive
2. Make it publicly accessible
3. Get direct download link
4. Download in `model_loader.py`:
   ```python
   import gdown
   url = "https://drive.google.com/uc?id=FILE_ID"
   gdown.download(url, 'best_f1_0.778.pt', quiet=False)
   ```

### Option 2: AWS S3
1. Upload to S3 bucket
2. Make it publicly readable
3. Download in startup script
   ```bash
   curl https://your-bucket.s3.amazonaws.com/best_f1_0.778.pt -o best_f1_0.778.pt
   ```

### Option 3: Hugging Face Hub
1. Upload model to Hugging Face
2. Download using `huggingface_hub`
   ```python
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(repo_id="username/repo", filename="model.pt")
   ```

---

## Next Steps After Deployment

1. ✅ Test with various images
2. ✅ Monitor performance and errors
3. ✅ Set up custom domain (optional)
4. ✅ Enable HTTPS (automatic on Render)
5. ✅ Configure health checks
6. ✅ Set up alerts for downtime

---

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
- GitHub Issues: https://github.com/SUGANIDHI/Project/issues

---

**Your services will be available at:**
- Backend: `https://stripunet-backend.onrender.com`
- Frontend: `https://stripunet-frontend.onrender.com`
- API Docs: `https://stripunet-backend.onrender.com/docs`

Good luck with your deployment! 🚀
