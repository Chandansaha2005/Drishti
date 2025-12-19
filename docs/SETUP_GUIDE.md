# DRISTI - Lost Person Detection System
# Setup and Configuration Guide

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Add CCTV Videos**
Place your video files in the `CCTVS/` folder:
```
CCTVS/
├── camera_1.mp4
├── camera_2.mp4
├── camera_3.mp4
└── camera_4.mp4
```

3. **Run the Application**
```bash
python main.py
```

4. **Open in Browser**
```
http://localhost:8000
```

## System Architecture

### Backend (main.py)
- FastAPI web server
- Handles photo uploads via `/api/search` endpoint
- Manages search job orchestration
- Serves frontend files
- Returns results via `/api/search-results/{search_id}`

### Face Search Service (search_service.py)
- Detects faces in uploaded photo using MediaPipe
- Extracts face embeddings using ResNet18
- Processes CCTV videos frame-by-frame
- Compares embeddings with cosine similarity
- Returns matches with confidence scores

### Frontend (Frontend/index.html)
- Drag-and-drop photo upload interface
- Real-time search progress indication
- Results display with thumbnails
- Camera location and timestamp info
- No authentication required

## Key Features

✅ **Single Page Application** - All functionality on one page
✅ **No Login** - Direct access to search
✅ **Multi-Video Support** - Search all cameras simultaneously
✅ **Accurate Detection** - MediaPipe + ResNet18 combination
✅ **Confidence Scores** - Shows match reliability
✅ **Timestamp Data** - Exact location in video where found
✅ **Evidence Snapshots** - Visual proof from CCTV footage

## API Flow

```
Frontend                          Backend                    Search Service
  |                                  |                               |
  |-- Upload Photo (POST /search)--->|                               |
  |                                  |-- Extract Face Embedding ----->|
  |                                  |<-- Embedding Returned ---------|
  |                                  |                               |
  |                                  |-- Process Each Video -------->|
  |                                  |   - Detect Faces              |
  |                                  |   - Compare Embeddings        |
  |                                  |   - Save Snapshots            |
  |                                  |<-- Matches Returned -----------|
  |                                  |                               |
  |-- Poll Results (GET /search-    |-- Save Results to JSON        |
  |    results/{id})--->|            |                               |
  |<-- Return Matches ----|          |                               |
  |                                  |                               |
Display Results with Images & Info
```

## Configuration Options

### Adjust Sensitivity
Edit `search_service.py`:

```python
# Lower = finds more matches (more false positives)
# Higher = finds fewer, more confident matches
self.similarity_threshold = 0.6  # Change to 0.5 or 0.7

# Process every Nth frame
# Higher = faster but might miss people
self.frame_skip = 5  # Change to 3 for more accuracy or 10 for speed
```

### Change Server Port
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## File Organization

```
Main Directory (f:\PROJECTS\Drishti\)
├── main.py                 ← START HERE
├── search_service.py       ← Face detection logic
├── requirements.txt        ← Dependencies
├── SETUP_GUIDE.md          ← This file
│
├── Frontend/
│   ├── index.html          ← Main webpage (NO AUTH)
│   └── assets/
│
├── CCTVS/                  ← Add your videos here
│   ├── camera_1.mp4
│   ├── camera_2.mp4
│   ├── camera_3.mp4
│   └── camera_4.mp4
│
├── uploads/                ← Auto-created: user photos
├── results/                ← Auto-created: search results
├── logs/                   ← Auto-created: application logs
└── models/                 ← Auto-created: ML models
```

## Workflow Example

1. User opens http://localhost:8000
2. User drags a photo of lost person into upload area
3. Frontend shows preview of photo
4. User clicks "Search CCTV Footage"
5. Backend receives photo and extracts face embedding
6. Backend loads all videos from CCTVS/ folder
7. For each video:
   - Reads frames
   - Detects all faces
   - Compares with lost person's face
   - If match found (confidence > threshold):
     - Saves snapshot with green box
     - Records camera name, time, confidence
8. Returns top matches sorted by confidence
9. Frontend displays matches with images and details

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | Fast, modern Python API |
| Web Server | Uvicorn | ASGI server for FastAPI |
| Video Processing | OpenCV | Read and process video files |
| Face Detection | MediaPipe | Fast face localization |
| Face Recognition | PyTorch + ResNet18 | Face embedding extraction |
| Image Processing | Pillow | Image manipulation |
| Computing | NumPy | Numerical operations |

## Performance Tips

- **Faster Search**: Increase `frame_skip` from 5 to 10
- **More Accurate**: Decrease `frame_skip` from 5 to 3
- **Save Memory**: Use lower resolution videos
- **Parallel Processing**: Already implemented with async operations

## Common Issues & Solutions

**Issue**: "No face detected in photo"
- Solution: Try different photo of same person or higher quality image

**Issue**: "No CCTV videos found"
- Solution: Ensure videos are in `CCTVS/` folder and named *.mp4

**Issue**: Search takes too long
- Solution: Increase `frame_skip` value or use shorter videos

**Issue**: Too many false positives
- Solution: Increase `similarity_threshold` from 0.6 to 0.7

**Issue**: Not finding the person
- Solution: Decrease `similarity_threshold` from 0.6 to 0.5 or 0.4

**Issue**: Port 8000 already in use
- Solution: Run on different port: `uvicorn main:app --port 8001`

## Video Format Requirements

- **Supported**: MP4, AVI, MOV
- **Recommended**: MP4 (best compatibility)
- **Resolution**: Any (higher = more accurate but slower)
- **Frame Rate**: Any (24-30 FPS typical)
- **Duration**: Any (longer = more search time)

## What Happens to Data

1. Uploaded photo: Temporarily stored in `uploads/` folder
2. Search results: Stored as JSON in `results/` folder
3. Snapshots: Saved as JPG in `results/` folder
4. No external API calls or cloud uploads
5. Everything stays on your local machine

## Extending the System

### Add Real-time Camera Feeds
Modify `search_service.py` to support RTSP streams instead of static videos

### Add Database
- Store search history
- Track found persons
- Generate reports

### Add Notifications
- Email alerts when person found
- SMS notifications
- Push notifications to mobile app

### Improve Accuracy
- Use more advanced face models (FaceNet, ArcFace)
- Add facial landmarks matching
- Implement age/gender filtering

## Troubleshooting Steps

1. **Check Backend is Running**
   ```bash
   curl http://localhost:8000/api/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Check Videos Exist**
   ```bash
   dir CCTVS
   ```
   Should list your MP4 files

3. **Check Python Packages**
   ```bash
   pip list
   ```
   Should show: fastapi, opencv-python, torch, mediapipe, etc.

4. **Check Logs**
   ```bash
   dir logs/
   ```
   Review any error messages

5. **Test Single Video**
   Modify `search_service.py` to process only one video for debugging

## Video Examples

For testing, any crowded video will work:
- Shopping mall security footage
- Train station camera
- Crowded street footage
- Family gathering videos
- Any video with multiple faces

## API Documentation

Interactive docs available at: `http://localhost:8000/docs`

Full OpenAPI spec: `http://localhost:8000/openapi.json`

## Important Notes

⚠️ **No Authentication** - Anyone with server access can search
⚠️ **Local Processing** - Requires good CPU/GPU for fast results
⚠️ **Privacy** - Ensure compliance with local privacy laws
⚠️ **Accuracy** - Face recognition is not 100% accurate

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Add videos to `CCTVS/` folder
3. Run: `python main.py`
4. Open browser: `http://localhost:8000`
5. Upload a test photo and search!

---

**DRISTI - Advanced Lost Person Detection**
