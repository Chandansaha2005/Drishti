"""
DRISTI - Lost Person Detection System
A simplified system to find lost persons in CCTV video footage
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from datetime import datetime
from pathlib import Path
import json
import asyncio
from typing import List, Dict, Any

from .search_service import SearchService

# Create necessary directories
UPLOAD_DIR = Path("../data/uploads")
RESULTS_DIR = Path("../data/results")
VIDEOS_DIR = Path("../data/cctvs")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="DRISTI - Lost Person Detection",
    description="Find lost persons in CCTV footage",
    version="1.0.0"
)

# CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = SearchService()

# Health check
@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "DRISTI Lost Person Detection"
    }

# Get available cameras (video files)
@app.get("/api/cameras")
async def get_cameras():
    """Get list of available CCTV cameras (video files)"""
    cameras = []
    if VIDEOS_DIR.exists():
        for video_file in VIDEOS_DIR.glob("*.mp4"):
            cameras.append({
                "id": video_file.stem,
                "name": video_file.stem.replace("_", " ").title(),
                "path": str(video_file),
                "filename": video_file.name
            })
    
    return {
        "success": True,
        "cameras": cameras,
        "total": len(cameras)
    }

# Upload lost person photo and search
@app.post("/api/search")
async def search_lost_person(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a photo of the lost person and search in all CCTV videos
    
    Returns:
    - Confidence percentage
    - Camera locations where person was found
    - Snapshots of matches
    """
    try:
        # Validate file
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file provided"}
            )
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / f"lost_person_{datetime.now().timestamp()}.jpg"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create search job ID
        search_id = f"search_{datetime.now().timestamp()}"
        
        # Get all video files
        videos = list(VIDEOS_DIR.glob("*.mp4"))
        
        if not videos:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No CCTV video files found",
                    "search_id": search_id
                }
            )
        
        # Run search in background
        if background_tasks:
            background_tasks.add_task(
                search_service.search_in_videos,
                str(upload_path),
                videos,
                search_id
            )
        else:
            # Run synchronously if no background tasks
            results = await search_service.search_in_videos(
                str(upload_path),
                videos,
                search_id
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "search_id": search_id,
                    "results": results,
                    "message": "Search completed"
                }
            )
        
        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "search_id": search_id,
                "message": "Search started. Results will be available shortly.",
                "status_url": f"/api/search-results/{search_id}"
            }
        )
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Search error: {str(e)}"}
        )

# Get search results
@app.get("/api/search-results/{search_id}")
async def get_search_results(search_id: str):
    """Get results from a completed search"""
    try:
        results_file = RESULTS_DIR / f"{search_id}.json"
        
        if not results_file.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Search not found or still processing",
                    "search_id": search_id
                }
            )
        
        with open(results_file, "r") as f:
            results = json.load(f)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "search_id": search_id,
                "results": results
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error retrieving results: {str(e)}"}
        )

# Get snapshot
@app.get("/api/snapshot/{filename}")
async def get_snapshot(filename: str):
    """Get snapshot image from search results"""
    try:
        snapshot_path = RESULTS_DIR / filename
        
        if not snapshot_path.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Snapshot not found"}
            )
        
        return FileResponse(snapshot_path, media_type="image/jpeg")
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Mount static files (frontend)
try:
    app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
except Exception as e:
    print(f"Warning: Could not mount frontend: {e}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("DRISTI - Lost Person Detection System")
    print("=" * 70)
    print("Starting server at http://localhost:8000")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Videos directory: {VIDEOS_DIR}")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
