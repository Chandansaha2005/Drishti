# DRISHTI - Lost Person Detection System

A comprehensive face recognition system that identifies lost persons in CCTV footage using AI-powered face detection and matching.

## Features

- 🔍 **Real-time Face Detection** - Detect faces in video streams using MediaPipe
- 🎯 **Face Matching** - Compare uploaded photos with detected faces
- 📹 **CCTV Integration** - Support for live RTSP streams from IP cameras
- 💾 **Local Video Support** - Process pre-recorded video files
- 📊 **Confidence Scoring** - Get matching confidence percentages
- 🎬 **Snapshot Capture** - Save matched faces for verification
- 🌐 **Web Interface** - Intuitive UI for searching lost persons

## System Architecture

![Architecture Diagram](docs/arch.png)

## Project Structure
```
DRISTI/
├── src/
│   ├── backend/
│   │   ├── app.py              # FastAPI application and routes
│   │   ├── search_service.py   # Face detection and matching logic
│   │   └── __init__.py
│   ├── config/
│   │   ├── settings.py         # Configuration settings
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
│
├── Frontend/                    # Web UI (HTML, CSS, JavaScript)
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── assets/
│
├── CCTVS/                       # CCTV video storage
│   ├── camera1.mp4
│   ├── camera2.mp4
│   └── ...
│
├── data/
│   ├── uploads/                # Temporary photo uploads
│   ├── results/                # Search results and snapshots
│   └── cctvs/                  # Alternative video location
│
├── docs/
│   ├── ARCHITECTURE.md         # Technical architecture
│   ├── QUICK_START.md          # Quick start guide
│   └── QUICK_START_CCTV.md     # CCTV setup guide
│
├── run.py                       # Application entry point
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```


## System Interface

### Main UI Views

<div align="center">

| CCTV Interface | Search Results |
|---|---|
| ![CCTV UI](Frontend/assets/cctv_ui.png) | ![Result UI](Frontend/assets/result_ui.png) |

</div>

### System Process Flow

<div align="center">

|  |  |
|---|---|
| ![Process 1](Frontend/assets/process1.png) | ![Process 2](Frontend/assets/process2.png) |
| ![Process 3](Frontend/assets/process3.png) | ![Process 4](Frontend/assets/process4.png) |

</div>

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) IP CCTV camera with RTSP support

### Setup Steps

1. **Clone/Download the Project**
   ```bash
   cd DRISHTI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python run.py
   ```
   The server will start at `http://localhost:8000`

4. **Access the Web Interface**
   Open your browser and go to:
   ```
   http://localhost:8000
   ```

## Usage

### Method 1: Using Local Video Files

1. **Prepare Videos**: Place MP4 video files in the `CCTVS/` folder
2. **Upload Photo**: Upload a photo of the lost person
3. **Search**: Click "Search CCTV Footage"
4. **View Results**: See matches with confidence scores

### Method 2: Using Live CCTV Cameras

#### Step 1: Add CCTV Cameras
1. Click the **"Connect with CCTV"** button (top right)
2. Follow the setup instructions to find your camera RTSP URL
3. Enter camera details:
   - **Camera Name**: e.g., "Main Entrance"
   - **Location**: e.g., "Gate A"
   - **RTSP URL**: e.g., `rtsp://username:password@192.168.1.100:554/stream1`
   - **Capture Duration**: 5-60 seconds (default 15)

#### Step 2: Test Connection
- Click **"Test Connection"** to verify RTSP access
- A message will confirm testing during actual search

#### Step 3: Save Configuration
- Click **"Save Configuration"** to activate CCTV
- Cameras will be stored locally in your browser

#### Step 4: Search
1. Upload lost person photo
2. Click **"Search CCTV Footage"**
3. System will capture 15 seconds from each camera
4. View results with match confidence

## Finding CCTV RTSP URLs

### Common CCTV Brands & Default URLs

| Brand | RTSP URL Format |
|-------|-----------------|
| **Hikvision** | `rtsp://username:password@ip:554/Streaming/Channels/101` |
| **Dahua** | `rtsp://username:password@ip:554/stream/main` |
| **Uniview** | `rtsp://username:password@ip:554/stream/profile1` |
| **Axis** | `rtsp://username:password@ip/axis-media/media.amp` |
| **Reolink** | `rtsp://username:password@ip:554/h264Preview_01_main` |
| **TP-Link** | `rtsp://username:password@ip:554/stream1` |
| **Generic** | `rtsp://username:password@ip:port/stream_path` |

### Steps to Find Your Camera's RTSP URL

1. **Find Camera IP Address**
   - Access your router settings (usually `192.168.1.1`)
   - Look in connected devices list
   - Or check camera sticker for default IP

2. **Access Camera Web Interface**
   - Open browser: `http://camera-ip`
   - Default login: `admin` / `12345` (check camera manual)

3. **Locate RTSP Settings**
   - Go to "Settings" or "Network" section
   - Look for "Streaming", "Stream", or "RTSP" option
   - Copy the RTSP URL

4. **Format the URL**
   ```
   rtsp://username:password@camera-ip:port/stream-path
   ```

## API Endpoints

### Search for Lost Person
```
POST /api/search
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPG, PNG)
- use_cctv: Boolean (optional)

Response:
{
  "success": true,
  "search_id": "search_1766172342.610262",
  "status_url": "/api/search-results/search_1766172342.610262"
}
```

### Get Search Results
```
GET /api/search-results/{search_id}

Response:
{
  "success": true,
  "search_id": "search_1766172342.610262",
  "results": {
    "status": "success",
    "matches": [
      {
        "camera": "cctv1",
        "camera_name": "Main Entrance",
        "confidence": 92.5,
        "frame_number": 1250,
        "time": 41.67,
        "time_formatted": "00:41",
        "snapshot": "snapshot_1234567890.jpg"
      }
    ]
  }
}
```

### Get Available Cameras
```
GET /api/cameras

Response:
{
  "success": true,
  "cameras": [
    {
      "id": "cctv1",
      "name": "Main Entrance",
      "type": "local",
      "location": "Gate A"
    }
  ]
}
```

## Configuration Files

### CCTV Config (Stored Locally)
Connected CCTV cameras are saved in browser localStorage:
- Key: `connectedCameras`
- Format: JSON array of camera objects

### Results
Search results are saved in `results/` folder:
- Filename: `search_{timestamp}.json`
- Contains all matches with snapshots

## Performance Tips

1. **Optimize Capture Duration**: Use 10-15 seconds for quick searches
2. **Camera Resolution**: Higher resolution = slower processing but better accuracy
3. **Similarity Threshold**: Current threshold is 0.6 (configurable in search_service.py)
4. **Frame Skip**: System processes every 5th frame for speed

## Troubleshooting

### RTSP Connection Fails
- ✅ Check camera IP is reachable: `ping camera-ip`
- ✅ Verify RTSP URL format includes username:password
- ✅ Ensure camera has RTSP enabled in settings
- ✅ Check firewall rules allow port 554

### No Faces Detected
- ✅ Ensure video quality is good
- ✅ Check lighting conditions
- ✅ Face must be clearly visible
- ✅ Try with higher confidence threshold

### Slow Processing
- ✅ Reduce video resolution
- ✅ Decrease capture duration
- ✅ Close other applications
- ✅ Check available disk space

## API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 202 | Search started (async) |
| 400 | Bad request |
| 404 | Resource not found |
| 500 | Server error |

## System Requirements

- **CPU**: Intel i5 or equivalent (GPU recommended for real-time)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 2GB minimum for videos
- **Network**: For CCTV: stable connection to camera IP

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- WebM (.webm)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Maximum size: 10MB

## Advanced Configuration

Edit `main.py` to customize:

```python
# Search service settings
SIMILARITY_THRESHOLD = 0.6  # 0.0-1.0 (lower = more matches)
FRAME_SKIP = 5              # Process every Nth frame
CAPTURE_DURATION_MAX = 15   # Maximum seconds to capture from CCTV
```

## Limitations

- Real-time processing limited by system resources
- RTSP capture limited to 15 seconds per camera
- Maximum 20 results displayed per search
- Accuracy depends on face visibility and lighting

## Future Enhancements

- [ ] Multiple face detection in single image
- [ ] Real-time streaming display
- [ ] Advanced filtering (age, gender, clothing)
- [ ] Database integration for historical searches
- [ ] Mobile app support
- [ ] Cloud deployment

## License

Specify your license here

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `logs/` folder
3. Check API response codes and error messages

## Credits

Built with:
- FastAPI
- MediaPipe
- OpenCV
- PyTorch
- Uvicorn

---

**Made By**: Subhojyoti Maity & Chandan Saha
