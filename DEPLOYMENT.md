# DRISHTI Deployment Guide: Render + Vercel

## Overview

This document provides complete, production-ready instructions for deploying DRISHTI on Render (backend) and Vercel (frontend). All configuration files required for deployment have been added to the repository.

---

## Part 1: Backend Deployment on Render

### Prerequisites

- Render account ([render.com](https://render.com))
- GitHub repository with this code
- Basic understanding of environment variables

### Project Structure Requirements

The backend expects this folder structure:

```
project-root/
├── run.py (entry point)
├── Procfile (Render build specification)
├── runtime.txt (Python version)
├── requirements.txt (dependencies)
├── src/
│   ├── backend/
│   │   ├── app.py (FastAPI application)
│   │   └── search_service.py
│   └── config/
│       └── settings.py (configuration)
├── data/
│   ├── uploads/ (user uploads)
│   ├── results/ (search results)
├── logs/ (application logs)
└── CCTVS/ (CCTV video storage)
```

All directories listed in `src/config/settings.py` will be created automatically at runtime.

### Required Python Version

**Python 3.11.7**

This is specified in `runtime.txt`. Render automatically reads this file.

### Critical Build & Start Commands

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn src.backend.app:app --worker-class uvicorn.workers.UvicornWorker --timeout 120
```

**Alternative (using Procfile):**
If you use a Procfile, Render will ignore custom build/start commands and use:
```
web: python run.py
```

**Recommendation:** Use the Procfile approach for simplicity. The `run.py` script automatically uses the `PORT` environment variable.

### Port Configuration

**CRITICAL:** Render dynamically assigns a PORT environment variable (typically 10000-10999).

The following files already respect this:
- `src/config/settings.py` → Reads `PORT` env variable (default: 8000)
- `run.py` → Uses `Settings.PORT` from configuration
- `src/backend/app.py` → Uses `Settings.PORT` when run directly

**Do NOT hardcode port 8000 in your start command.** The code is already fixed to use `os.getenv('PORT', 8000)`.

### Environment Variables (Backend)

Set these in the Render dashboard under **Environment**:

| Variable | Purpose | Usage | Required | Example |
|---|---|---|---|---|
| `PORT` | HTTP server port (auto-set by Render) | `src/config/settings.py` | Optional* | *Auto-assigned* |
| `DEBUG` | Enable debug mode | `src/config/settings.py` | Optional | `False` |
| `DATABASE_URL` | SQLite or PostgreSQL connection | `src/config/settings.py` | Optional | `sqlite:///./dristi.db` |
| `SECRET_KEY` | JWT secret for auth (future use) | `src/config/settings.py` | Optional | `your-secret-key-here` |
| `SIMILARITY_THRESHOLD` | Face match confidence (0-1) | `src/config/settings.py` | Optional | `0.60` |

*PORT is automatically assigned by Render and does not need to be set manually.

### Required Dependencies

All dependencies are in `requirements.txt`:

- **fastapi==0.104.1** - Web framework
- **uvicorn==0.24.0** - ASGI server
- **torch==2.5.1** - Deep learning (PyTorch) - Large file, may increase cold start
- **torchvision==0.20.1** - Computer vision
- **mediapipe==0.10.21** - Face detection and recognition
- **opencv-python==4.8.1.78** - Video/image processing
- **python-multipart==0.0.6** - Form data parsing
- **python-dotenv==1.0.0** - Environment variable loading
- Additional: numpy, Pillow, matplotlib, aiofiles

### Deployment Steps (Render)

1. **Connect GitHub Repository**
   - Go to [render.com/dashboard](https://render.com/dashboard)
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Select the branch to deploy

2. **Configure Service**
   - **Service Name:** `drishti-backend` (or your choice)
   - **Environment:** `Python 3`
   - **Region:** Choose closest to your users
   - **Branch:** `main` (or your main branch)
   - **Build Command:** Leave empty (Procfile takes precedence)
   - **Start Command:** Leave empty (Procfile takes precedence)

3. **Add Environment Variables**
   - Click "Add Environment Variable"
   - Add `DEBUG=False`SECRET_KEY=<your-random-secret-key>
   - Add ``
   - Do NOT add PORT (Render sets this automatically)

4. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy automatically
   - Monitor logs in the Render dashboard

5. **Verify Deployment**
   - Copy your service URL (e.g., `https://drishti-backend-xxxxx.onrender.com`)
   - Test health endpoint: `curl https://drishti-backend-xxxxx.onrender.com/api/health`
   - Expected response: `{"status":"healthy","timestamp":"...","system":"DRISTI Lost Person Detection v2.0"}`

### Render-Specific Caveats

**Cold Start Delay:**
- First request after inactivity takes 30-60 seconds
- Torch/MediaPipe imports are heavy and slow on first load
- This is expected and cannot be avoided on Render's free tier

**File System:**
- `/tmp` is available and writable (persists for current deployment)
- All other writes may be ephemeral; use database or cloud storage for persistence
- `data/uploads` and `data/results` directories are created in `/tmp`
- **Important:** On free tier, files may be deleted when dyno sleeps

**Memory:**
- Render free tier: 0.5 GB RAM
- Torch + MediaPipe + OpenCV consume ~800 MB
- May face OOM on complex operations
- For production, upgrade to paid tier (2+ GB)

**CPU:**
- Single core on free tier
- Video processing is CPU-intensive
- Consider upgrading for real-world use

**Build Time:**
- First build: 5-10 minutes (installing Torch)
- Subsequent builds: 2-3 minutes

---

## Part 2: Frontend Deployment on Vercel

### Frontend Structure

```
Frontend/
├── index.html (main page)
├── script.js (UI logic)
├── api.js (API client)
├── auth.js (authentication logic)
├── style.css (styling)
├── vercel.json (Vercel config)
├── .env.production (production env vars)
└── assets/
    └── (images, icons, etc)
```

### Deployment Steps (Vercel)

1. **Connect GitHub Repository**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New..."
   - Select "Project"
   - Import your GitHub repository
   - Select the repository containing DRISHTI

2. **Configure Project Settings**
   - **Framework Preset:** `Other` (it's a static site, not Next.js)
   - **Root Directory:** `Frontend`
   - **Build Command:** Leave empty (not needed for static HTML)
   - **Output Directory:** Leave empty
   - **Install Command:** `echo 'No installation needed'`

3. **Add Environment Variables**
   - Click "Environment Variables"
   - Add new variable: `VITE_API_BACKEND_URL`
   - Value: Your Render backend URL (e.g., `https://drishti-backend-xxxxx.onrender.com`)
   - **Important:** Environment variables are NOT injected into static HTML
   - See "Configuring Backend URL" section below

4. **Deploy**
   - Click "Deploy"
   - Vercel automatically deploys to a unique URL
   - Each push to `main` triggers automatic redeployment

5. **Access Your Frontend**
   - Vercel provides a URL: `https://your-project.vercel.app`
   - Configure custom domain in Vercel settings if desired

### Configuring Backend API URL

**Current Method (Dynamic):**

The frontend (`Frontend/api.js`) now auto-detects the backend URL:

```javascript
export const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : (window.API_BACKEND_URL || 'https://drishti-backend-ncjd.onrender.com');
```

**For Production on Vercel:**

1. **Option A: Inject URL via Window Object (Recommended)**

   In your Vercel deployment, add a build step that generates a config script:
   
   Create `Frontend/config.js`:
   ```javascript
   window.API_BACKEND_URL = 'https://drishti-backend-xxxxx.onrender.com';
   ```
   
   Add this to `Frontend/index.html` (before other scripts):
   ```html
   <script src="./config.js"></script>
   ```

2. **Option B: Update API_BASE_URL Manually**

   After deploying to Vercel, edit `Frontend/api.js` with your actual Render URL:
   ```javascript
   export const API_BASE_URL = 'https://drishti-backend-xxxxx.onrender.com';
   ```

3. **Option C: Use Environment Variables with Build Step (Advanced)**

   Vercel environment variables are NOT available in static HTML. For this to work, you would need to:
   - Add a build step that processes files
   - Use a simple build tool (like a Node script)
   - Not recommended for this project's simplicity

**Recommended:** Use Option A (inject via `config.js`).

### Frontend Environment Variables

Since this is a static site, environment variables do NOT automatically inject into HTML.

- **`VITE_API_BACKEND_URL`** - Set in Vercel dashboard, but it won't inject automatically
- **Solution:** Use the `window.API_BACKEND_URL` method described above

### Testing Frontend Locally

1. Start the backend:
   ```bash
   python run.py
   # or
   uvicorn src.backend.app:app --reload
   ```

2. Open `Frontend/index.html` in a browser:
   ```bash
   cd Frontend
   python -m http.server 8080
   # Then visit http://localhost:8080
   ```

3. The frontend will use `http://localhost:8000` for API calls (auto-detected)

---

## Part 3: Common Deployment Errors & Solutions

### Error 1: Port Binding Failed

**Symptom:**
```
RuntimeError: Error binding to address 0.0.0.0:8000: Address already in use
```

**Cause:**
- Hardcoded port conflicts with Render's assigned port
- Another process using that port locally

**Solution:**
- ✓ Code is already fixed to use `Settings.PORT` and env variable `PORT`
- Ensure `run.py` includes: `port = int(os.getenv('PORT', 8000))`
- On Render, never set custom start command that hardcodes port

### Error 2: Python Version Mismatch

**Symptom:**
```
ModuleNotFoundError: No module named 'module_x'
ModuleNotFoundError: No module named 'torch'
```

**Cause:**
- Torch not available in Python 3.12+ (Render uses 3.11+ by default)
- requirements.txt uses versions incompatible with Python version

**Solution:**
- ✓ Code includes `runtime.txt` specifying Python 3.11.7
- Render reads `runtime.txt` and uses exact Python version
- If issue persists, verify `runtime.txt` exists with `3.11.7`

### Error 3: Torch Import Fails on Render

**Symptom:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Cause:**
- PyTorch needs OpenGL libraries not available in Render's container
- MediaPipe and OpenCV may have missing system dependencies

**Solution:**
- Use pre-compiled Torch wheel from PyPI (included in requirements.txt)
- Ensure requirements.txt specifies exact versions (already done)
- On Render, Torch installs from PyPI binary wheels (should work)
- If fails, try: `torch==2.0.0` (slightly older, broader compatibility)

### Error 4: MediaPipe Initialization Fails

**Symptom:**
```
RuntimeError: tensorflow/lite/kernels/register.cc:...
ResourceExhaustedError: OOM when allocating tensor
```

**Cause:**
- MediaPipe model loading exceeds available RAM (Render free tier: 0.5 GB)
- Too many concurrent requests

**Solution:**
- Solution 1: Limit concurrent uploads (queue requests server-side)
- Solution 2: Upgrade Render plan to 2+ GB RAM
- Solution 3: Reduce model complexity (not feasible for this project)

### Error 5: Frontend Cannot Connect to Backend

**Symptom:**
```
CORS error in browser console
fetch() fails with "Failed to fetch"
```

**Cause:**
- Backend URL is hardcoded to `localhost:8000` or wrong URL
- CORS not enabled on backend
- Backend not running or URL is incorrect

**Solution:**
- ✓ CORS is already enabled in `src/backend/app.py`:
  ```python
  app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
  ```
- Update frontend API_BASE_URL to actual Render URL
- Test backend health: `curl https://drishti-backend-xxxxx.onrender.com/api/health`

### Error 6: Cold Start Takes Too Long

**Symptom:**
- First request to deployed backend times out (>120 seconds)

**Cause:**
- Torch initialization is slow
- Render's free tier resources are limited

**Solution:**
- ✓ This is expected behavior
- Solution 1: Use Render's "Keep Alive" (background job) to prevent cold starts
- Solution 2: Upgrade to paid Render plan
- Solution 3: Cache models or use model optimization

### Error 7: File Uploads Not Persisting

**Symptom:**
- Files uploaded to `data/uploads` disappear after deployment
- Search results in `data/results` are lost

**Cause:**
- Render free tier has ephemeral file system
- Files written to disk are not persisted across dyno restarts

**Solution:**
- ✓ For development: Ephemeral storage is acceptable
- For production: Use cloud storage
  - Option 1: AWS S3 + boto3
  - Option 2: Google Cloud Storage
  - Option 3: Azure Blob Storage
  - Option 4: Upgrade Render to paid plan with persistent disk
- Modify `src/backend/app.py` to use cloud storage instead of local disk

### Error 8: CCTVS Directory Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'CCTVS'
```

**Cause:**
- `CCTVS/` directory doesn't exist on Render

**Solution:**
- ✓ Code automatically creates missing directories via `Settings.create_directories()`
- If error persists, check that `src/config/settings.py` is being imported
- Manually create `CCTVS/` directory in repository (optional)

### Error 9: OpenCV Codec Errors

**Symptom:**
```
OpenCV(4.8.1) error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
```

**Cause:**
- Video codec not supported or file corrupted
- Container doesn't have required multimedia libraries

**Solution:**
- Ensure video files in `CCTVS/` are valid MP4/AVI/MOV
- Install system-level ffmpeg (usually available on Render)
- Add to start command if needed: `apt-get install -y ffmpeg` (not standard on Render)

---

## Part 4: Verification Checklist

Use this checklist AFTER deployment to verify everything works:

### Backend (Render)

- [ ] **Health Check**
  - Endpoint: `GET https://drishti-backend-xxxxx.onrender.com/api/health`
  - Expected: `{"status":"healthy","timestamp":"...","system":"DRISTI..."}`
  - Run: `curl https://drishti-backend-xxxxx.onrender.com/api/health`

- [ ] **Port Binding Correct**
  - Backend should respond on Render's assigned port (NOT hardcoded 8000)
  - Check Render dashboard logs: should show port assignment

- [ ] **CORS Enabled**
  - Frontend should be able to make requests to backend
  - Check browser DevTools → Network tab
  - Requests should NOT show "CORS error"

- [ ] **File Uploads Work**
  - Test: Upload an image via frontend
  - Check backend logs (Render dashboard)
  - Files should be saved to `data/uploads/`

### Frontend (Vercel)

- [ ] **Page Loads**
  - URL: `https://your-project.vercel.app`
  - Page should load and display DRISHTI UI

- [ ] **API Client Configured**
  - Check browser console (F12)
  - Check `api.js` API_BASE_URL points to your Render backend
  - No errors like "Cannot find module 'api.js'"

- [ ] **Backend Connection Works**
  - Open browser DevTools → Network tab
  - Click "Search" button (or similar)
  - Should make request to your Render backend URL
  - Check response status (should be 200, 201, or 400+, not network error)

### Integration Test

- [ ] **Full Flow**
  1. Open frontend URL
  2. Upload an image
  3. Click search
  4. Backend should process request
  5. Results should display
  6. Check logs: `Render dashboard` → service logs

- [ ] **Search Returns Results**
  - Backend should find matches or return "no matches found"
  - Response time may be 30-120 seconds (especially on cold start)
  - Status should be visible in UI

- [ ] **Error Handling**
  - Try uploading invalid file
  - Frontend should show error message
  - Backend should reject with 400 status

### Logs to Check on Failure

**Render (Backend Logs):**
```
Render dashboard → Your service → Logs tab
```
Look for:
- `ModuleNotFoundError` → missing dependency
- `Port already in use` → port binding issue
- `OOM` → out of memory
- `CORS error in logs` → CORS configuration issue

**Vercel (Frontend Logs):**
```
Vercel dashboard → Project → Deployments → Logs
```
Look for:
- Build errors (should be minimal for static site)
- Deploy errors

**Browser Console (Frontend):**
```
Open deployed frontend → Press F12 → Console tab
```
Look for:
- `Failed to fetch` → backend not accessible
- `CORS error` → CORS misconfiguration
- `API_BASE_URL is undefined` → environment variable not set

---

## Summary: Deployment Readiness

This project is now fully deployable:

**Changes Made:**
1. ✓ `run.py` - Updated to use `PORT` env variable
2. ✓ `src/backend/app.py` - Updated to use `Settings.PORT`
3. ✓ `Procfile` - Created for Render deployment
4. ✓ `runtime.txt` - Specified Python 3.11.7
5. ✓ `Frontend/api.js` - Updated to use dynamic API_BASE_URL
6. ✓ `Frontend/vercel.json` - Created for Vercel config
7. ✓ `Frontend/.env.production` - Created for production env vars
8. ✓ `.env.example` - Updated with all env variables

**Next Steps:**
1. Push code to GitHub
2. Deploy backend to Render (see Part 1)
3. Deploy frontend to Vercel (see Part 2)
4. Configure backend URL in frontend
5. Run verification checklist above

**Expected Costs:**
- Render free tier: $0 (limitations: 0.5 GB RAM, cold starts, ephemeral storage)
- Vercel free tier: $0 (limitations: bandwidth caps, but sufficient for most uses)
- Both offer paid tiers for production workloads

