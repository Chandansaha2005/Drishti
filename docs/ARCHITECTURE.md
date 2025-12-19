# DRISTI - System Architecture & Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Deployment](#deployment)

---

## System Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│                      (Frontend - index.html)                     │
│  - Upload drag-and-drop interface                               │
│  - Real-time search progress                                    │
│  - Results display with thumbnails                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                               │
│                      (main.py)                                   │
│  - POST /api/search - Accept photo upload                       │
│  - GET /api/search-results/{id} - Poll for results              │
│  - GET /api/snapshot/{filename} - Serve match images            │
│  - GET /api/cameras - List available videos                     │
│  - GET /api/health - Health check                               │
│  - Static file serving (Frontend)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │ Calls
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               Search Service                                     │
│              (search_service.py)                                 │
│  - Face Detection (MediaPipe)                                   │
│  - Face Embedding (ResNet18)                                    │
│  - Video Processing (OpenCV)                                    │
│  - Similarity Calculation (NumPy)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │ Reads/Writes
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   File System                                    │
│  ├── CCTVS/ - Input video files                                 │
│  ├── uploads/ - Temporary photo storage                         │
│  └── results/ - Search results (JSON + JPG snapshots)           │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML/CSS/JavaScript | User interface |
| Backend | FastAPI + Uvicorn | Web server & API |
| Video | OpenCV | Read and process videos |
| Detection | MediaPipe | Detect faces in frames |
| Recognition | ResNet18 + PyTorch | Extract face features |
| Computing | NumPy | Numerical operations |
| Imaging | Pillow | Image manipulation |

---

## Component Details

### 1. Frontend (Frontend/index.html)

**Purpose**: User interface for uploading photos and viewing results

**Key Features**:
- Drag-and-drop photo upload
- Preview of selected photo
- Real-time search progress indication
- Results grid with match details
- No authentication required

**Key Functions**:
```javascript
// Upload handling
uploadArea.addEventListener('drop', (e) => handleFileSelect(e.dataTransfer.files[0]))

// Search initiation
startSearch()
  ├─ Validate file
  ├─ POST to /api/search
  ├─ Poll /api/search-results/{search_id}
  └─ Display results

// Result display
displayResults(results)
  ├─ Show summary stats
  ├─ Render match cards
  └─ Load snapshots
```

**Endpoints Called**:
- `POST /api/search` - Upload photo
- `GET /api/search-results/{search_id}` - Get results

---

### 2. Backend (main.py)

**Purpose**: HTTP server and request handling

**Key Components**:

#### Initialization
```python
app = FastAPI()
search_service = SearchService()
```

#### Routes

**GET /api/health**
- Returns: `{"status": "healthy", ...}`
- Purpose: System health check

**GET /api/cameras**
- Returns: List of available CCTV cameras
- Purpose: Inform frontend of available videos

**POST /api/search**
- Input: Multipart form with image file
- Process: Save file, initiate search
- Return: `{"search_id": "search_123...", "status_url": "..."}`
- Background: Async search processing

**GET /api/search-results/{search_id}**
- Input: Search ID
- Return: Complete results with matches
- Format: JSON with matches array

**GET /api/snapshot/{filename}**
- Input: Snapshot filename
- Return: JPEG image from results directory

---

### 3. Search Service (search_service.py)

**Purpose**: Core face detection and matching logic

**Class: SearchService**

#### Methods

**`_load_face_model()`**
- Loads pre-trained ResNet18
- Returns: Neural network model for face embeddings
- Error handling: Graceful fallback to histogram method

**`extract_face_embedding(face_image)`**
- Input: Face image (numpy array)
- Process:
  1. Resize to 224x224
  2. Normalize using ImageNet stats
  3. Pass through ResNet18
  4. Extract output from second-to-last layer
  5. L2 normalize embedding
- Output: Embedding vector (512-dim or fallback 256-dim)

**`detect_faces_in_frame(frame)`**
- Input: Video frame (numpy array)
- Process:
  1. Convert BGR to RGB
  2. Use MediaPipe for detection
  3. Extract bounding boxes
  4. Add padding to boxes
  5. Extract face ROI
- Output: List of detected faces with bboxes and confidences

**`calculate_similarity(embedding1, embedding2)`**
- Input: Two face embeddings
- Process: Cosine similarity = dot(e1, e2) / (|e1| * |e2|)
- Output: Similarity score (0-1, higher = more similar)

**`search_in_videos(lost_person_path, video_paths, search_id)`**
- Main search function
- Steps:
  1. Load and extract embedding from lost person photo
  2. For each video file:
     - Open with OpenCV
     - For each frame (skip by FRAME_SKIP):
       - Detect faces
       - Extract embeddings
       - Compare with lost person
       - If match found:
         - Save snapshot
         - Record camera, time, confidence
  3. Sort results by confidence
  4. Save to JSON file
- Output: Complete search results

---

### 4. Video Processing Flow

```
Video File (.mp4)
      │
      ▼
┌─────────────┐
│  cv2.open   │
└──────┬──────┘
       │ frame_idx, frame_data
       ▼
┌──────────────────────────┐
│ Skip frames (FRAME_SKIP) │  (every 5th frame)
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ MediaPipe Face Detection │
└──────┬───────────────────┘
       │ detected_faces[]
       ▼
┌──────────────────────────┐
│ For each detected face:  │
│  1. Extract ROI          │
│  2. Get embedding        │
│  3. Calculate similarity │
└──────┬───────────────────┘
       │
       ▼ (if match > threshold)
┌──────────────────────────┐
│ Save Snapshot            │
│ Record Results           │
└──────┬───────────────────┘
       │
       ▼ (next frame)
    [repeat]
       │
       ▼ (all frames processed)
┌──────────────────────────┐
│ Return Matches           │
└──────────────────────────┘
```

---

## Data Flow

### Complete Search Workflow

```
1. USER UPLOADS PHOTO
   ├─ Browser: File selected
   ├─ Frontend: Validate file type/size
   ├─ Frontend: Show preview
   └─ Frontend: POST to /api/search with FormData

2. BACKEND RECEIVES FILE
   ├─ Save to uploads/
   ├─ Create search_id
   ├─ Get list of video files
   ├─ Launch async search task
   └─ Return: {"search_id": "...", "status_url": "..."}

3. SEARCH EXECUTES
   ├─ Extract lost person face embedding
   ├─ For each video:
   │  ├─ Open with OpenCV
   │  ├─ Read frame by frame
   │  ├─ Detect faces (MediaPipe)
   │  ├─ Extract embeddings (ResNet18)
   │  ├─ Compare (cosine similarity)
   │  ├─ If match found:
   │  │  ├─ Save snapshot JPG
   │  │  ├─ Record metadata
   │  │  └─ Add to results[]
   │  └─ Close video
   ├─ Sort results by confidence
   └─ Save to results/{search_id}.json

4. FRONTEND POLLS FOR RESULTS
   ├─ GET /api/search-results/{search_id}
   ├─ If not ready: 404, wait 1 second, retry
   ├─ If ready: Receive JSON response
   ├─ Parse results
   └─ Display matches

5. USER VIEWS RESULTS
   ├─ Show summary (total matches, best confidence)
   ├─ Load match cards in grid
   ├─ For each match:
   │  ├─ Display snapshot thumbnail
   │  ├─ Show camera name
   │  ├─ Show confidence %
   │  ├─ Show time in video
   │  └─ Show frame number
   └─ Sort by confidence (highest first)
```

### Data Format: Search Results JSON

```json
{
  "search_id": "search_1702968000.123",
  "timestamp": "2025-12-19T22:40:00",
  "status": "completed",
  "matches": [
    {
      "camera": "cctv1",
      "camera_name": "CCTV 1",
      "confidence": 92.5,
      "timestamp": 120.5,
      "frame_number": 25,
      "snapshot": "search_123_cctv1_25.jpg",
      "time_formatted": "02:00"
    },
    {
      "camera": "cctv2",
      "camera_name": "CCTV 2",
      "confidence": 87.3,
      "timestamp": 245.2,
      "frame_number": 51,
      "snapshot": "search_123_cctv2_51.jpg",
      "time_formatted": "04:05"
    }
  ],
  "stats": {
    "total_videos": 4,
    "videos_processed": 4,
    "total_frames_processed": 2845,
    "matches_found": 2
  },
  "summary": {
    "total_matches": 2,
    "best_match_confidence": 92.5,
    "cameras_with_matches": 2
  }
}
```

---

## API Reference

### 1. Health Check

```
GET /api/health
```

**Response (200)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-19T22:40:00",
  "system": "DRISTI Lost Person Detection"
}
```

### 2. Get Cameras

```
GET /api/cameras
```

**Response (200)**:
```json
{
  "success": true,
  "cameras": [
    {
      "id": "cctv1",
      "name": "Cctv 1",
      "path": "CCTVS/cctv1.mp4",
      "filename": "cctv1.mp4"
    }
  ],
  "total": 4
}
```

### 3. Upload and Search

```
POST /api/search
Content-Type: multipart/form-data

Body:
  file: [binary image data]
```

**Response (202 - Accepted)**:
```json
{
  "success": true,
  "search_id": "search_1702968000.12345",
  "message": "Search started. Results will be available shortly.",
  "status_url": "/api/search-results/search_1702968000.12345"
}
```

**Response (400 - Bad Request)**:
```json
{
  "success": false,
  "message": "No file provided"
}
```

### 4. Get Results

```
GET /api/search-results/{search_id}
```

**Response (200 - Found)**:
```json
{
  "success": true,
  "search_id": "search_...",
  "results": {
    "matches": [...],
    "stats": {...},
    "summary": {...}
  }
}
```

**Response (404 - Not Found)**:
```json
{
  "success": false,
  "message": "Search not found or still processing",
  "search_id": "search_..."
}
```

### 5. Download Snapshot

```
GET /api/snapshot/{filename}
```

**Response (200)**:
- Content-Type: image/jpeg
- Body: JPEG image binary data

---

## Configuration

### Tuning Parameters (search_service.py)

```python
# Face similarity threshold
# Higher = stricter (fewer matches)
# Lower = lenient (more matches)
# Recommended: 0.5-0.7
self.similarity_threshold = 0.6

# Frame skip
# Higher = faster but might miss faces
# Lower = slower but more thorough
# Recommended: 3-10
self.frame_skip = 5

# Face detection confidence
# Minimum confidence to consider a face detected
# Recommended: 0.5-0.7
min_detection_confidence=0.5
```

### Environment Variables (.env)

```
HOST=0.0.0.0
PORT=8000
DEBUG=False
LOG_LEVEL=INFO
```

---

## Deployment

### Local Development
```bash
python main.py
# Server runs at http://localhost:8000
```

### Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Performance Metrics

| Operation | Time | Resources |
|-----------|------|-----------|
| Load ResNet18 | 2-3 sec | 100 MB RAM |
| Extract embedding | 50 ms | 10 MB RAM |
| Detect faces (frame) | 30 ms | 20 MB RAM |
| Process 1-hour video | 3-5 min | 500 MB peak |
| All 4 videos (~12 min) | 15-20 min | 800 MB peak |

---

## Error Handling

### Frontend Errors
- File validation
- Network errors
- Timeout handling
- Result parsing errors

### Backend Errors
- Invalid file upload
- Missing video files
- Face detection failures
- Database/storage errors

### Recovery Strategies
- Automatic retries for polling
- Timeout fallbacks
- Graceful degradation
- Clear error messages to user

---

## Security Considerations

1. **Input Validation**
   - File type checking
   - File size limits
   - Path traversal prevention

2. **Output Security**
   - Filenames sanitized
   - Results directory protected
   - No sensitive data exposure

3. **Access Control**
   - No authentication (as required)
   - localhost only (production should add auth)

---

## Monitoring & Logging

### Log Locations
- `logs/face_recognition.log` - Face recognition operations
- `logs/search_service.log` - Search operations
- Console output - Real-time monitoring

### Key Metrics
- Searches performed
- Average confidence scores
- Videos processed
- Processing times
- Error rates

---

## Future Enhancements

1. **Real-time Streams**: RTSP camera feed support
2. **Database**: Store search history and results
3. **Notifications**: Email/SMS alerts on matches
4. **Analytics**: Dashboard with statistics
5. **Mobile**: Native iOS/Android apps
6. **Advanced ML**: FaceNet, ArcFace models
7. **Multi-person**: Search for multiple people
8. **Filtering**: Age/gender-based filtering

---

**DRISTI - Advanced Face Recognition for Public Safety**
