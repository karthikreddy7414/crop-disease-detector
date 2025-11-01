# 🚀 Render Deployment Guide

## Quick Fix for Your Deployment Issue

Your Render deployment was failing because the code was looking for wrong model filenames. This has been **FIXED** ✅

## What Was Wrong

1. **Filename Mismatch**: Code looked for `image_classifier_best.joblib` but lightweight models are named `image_classifier.joblib`
2. **Solution**: Updated `src/api/routes/image_processing.py` to check for correct filenames

## Deploy to Render - Step by Step

### Prerequisites
- GitHub account with your repository
- Render account (free tier works!)

### Step 1: Push Your Fixed Code

```bash
# Add all changes including the fix
git add .

# Commit the changes
git commit -m "Fix: Update model loading for Render deployment"

# Push to GitHub
git push origin main
```

### Step 2: Deploy on Render

1. Go to **https://render.com** and sign in with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your repository: `DNA-crop` or your repo name
4. Configure the following settings:

   **Basic Settings:**
   - **Name**: `crop-disease-detector` (or your preferred name)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: (leave blank)

   **Build & Deploy:**
   - **Build Command**: 
     ```
     pip install -r requirements-render.txt
     ```
   
   - **Start Command**: 
     ```
     uvicorn src.main:app --host 0.0.0.0 --port $PORT
     ```

   **Instance Type:**
   - **Free** (or paid if you prefer)

5. Click **"Create Web Service"**

### Step 3: Wait for Deployment

- Render will install dependencies (~3-5 minutes)
- Your app will be live at: `https://your-service-name.onrender.com`

### Step 4: Test Your Deployment

Visit your Render URL and test:
```
https://your-service-name.onrender.com/health
```

You should see:
```json
{"status": "healthy"}
```

## ⚙️ Current Model Configuration

Your deployment uses **lightweight models** (10 disease classes):

| Disease Class | Crop |
|---------------|------|
| Apple___Apple_scab | Apple |
| Apple___healthy | Apple |
| Corn_(maize)___Common_rust_ | Corn |
| Corn_(maize)___healthy | Corn |
| Grape___Black_rot | Grape |
| Grape___healthy | Grape |
| Potato___Early_blight | Potato |
| Potato___Late_blight | Potato |
| Potato___healthy | Potato |
| Pepper,_bell___healthy | Pepper |

**Accuracy**: 82.4%
**Model Size**: 0.54 MB (GitHub-friendly!)

## 🔧 Troubleshooting

### Issue: "Application failed to respond"

**Solution**: Check Render logs:
1. Go to your service dashboard
2. Click "Logs" tab
3. Look for errors

Common fixes:
- Ensure `PORT` environment variable is used correctly ✅ (already done)
- Check if all dependencies installed ✅ (requirements-render.txt is minimal)

### Issue: "Module not found"

**Solution**: Add missing package to `requirements-render.txt`

### Issue: Models not loading

**Solution**: Already fixed! The code now correctly loads lightweight models.

### Issue: Out of memory

**Solution**: Lightweight models are only 0.54 MB, this shouldn't happen. If it does:
1. Check Render instance type (Free tier has 512 MB RAM)
2. Reduce `WORKERS` in config (currently 4)

## 📊 API Endpoints

After deployment, your API will have:

- **GET** `/` - Web interface
- **GET** `/health` - Health check
- **POST** `/api/image/preprocess` - Disease prediction
- **GET** `/docs` - Interactive API documentation

## 🎯 Performance Tips

### Free Tier Limitations
- Spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- 512 MB RAM limit

### Upgrade to Paid Tier ($7/month) for:
- No spin-down
- Faster response times
- More memory
- Custom domain support

## 🔄 Updating Your Deployment

Whenever you make changes:

```bash
git add .
git commit -m "Your update message"
git push origin main
```

Render will automatically redeploy! 🎉

## ✅ Deployment Checklist

Before deploying, ensure:

- [x] Lightweight models exist in `models/lightweight/`
- [x] Models are tracked in Git (not in .gitignore)
- [x] `requirements-render.txt` has all dependencies
- [x] `runtime.txt` specifies Python version
- [x] Code loads correct model filenames
- [x] CORS is enabled for all origins
- [x] PORT environment variable is used
- [x] Start command uses `0.0.0.0` host

## 📝 Environment Variables (Optional)

You can set these in Render dashboard under "Environment":

- `ENVIRONMENT` = `production`
- `LOG_LEVEL` = `INFO`
- `WORKERS` = `2` (reduce if memory issues)

## 🆘 Need Help?

1. Check Render logs first
2. Verify models are in GitHub: `git ls-files models/lightweight/`
3. Test locally first: `python -m uvicorn src.main:app --port 8000`
4. Check this guide's troubleshooting section

## 🎉 Success!

Once deployed, share your link:
```
https://your-service-name.onrender.com
```

Your crop disease detection system is now live! 🌱

