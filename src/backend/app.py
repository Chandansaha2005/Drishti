"""
DRISTI - Lost Person Detection System
Main FastAPI application for face recognition in CCTV footage.
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
import sys
import asyncio

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.search_service import SearchService
from src.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Settings.LOG_DIR / 'drishti.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize directories
Settings.create_directories()

# Initialize FastAPI app
app = FastAPI(
    title="DRISTI - Lost Person Detection",
    description="Find lost persons in CCTV footage using face recognition",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = SearchService()


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "DRISTI Lost Person Detection v2.0"
    }


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.post("/api/search")
async def search_lost_person(
    file: UploadFile = File(...),
    use_cctv: bool = Query(False, description="Use connected CCTV cameras")
):
    """
    Upload a photo of a lost person and search in CCTV videos.
    Runs synchronously using asyncio.to_thread() to avoid Render timeout issues.
    
    Returns:
        - Results with matched faces
        - Confidence percentages
        - Timestamps and camera info
    """
    try:
        # Validate file
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file provided"}
            )
        
        # Save uploaded file
        upload_path = Settings.UPLOAD_DIR / f"lost_person_{datetime.now().timestamp()}.jpg"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create unique search ID
        search_id = f"search_{datetime.now().timestamp()}"
        
        # Get all available video files
        videos = list(Settings.VIDEO_DIR.glob("*.mp4"))
        
        if not videos:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No CCTV video files found in the system",
                    "search_id": search_id
                }
            )
        
        # Run search synchronously using asyncio.to_thread() to prevent timeout
        # This executes the CPU-heavy task in a thread pool without blocking the event loop
        logger.info(f"Starting face search for {len(videos)} video file(s)...")
        results = await asyncio.to_thread(
            search_service.search_in_videos,
            str(upload_path),
            videos,
            search_id
        )
        
        # Return results directly (status 200)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "search_id": search_id,
                "results": results,
                "message": "Search completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Search error: {str(e)}"}
        )


@app.get("/api/search-results/{search_id}")
async def get_search_results(search_id: str):
    """Retrieve search results by ID"""
    try:
        results_file = Settings.RESULTS_DIR / f"{search_id}.json"
        
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
        logger.error(f"Error retrieving results: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@app.get("/api/snapshot/{filename}")
async def get_snapshot(filename: str):
    """Retrieve snapshot image from search results"""
    try:
        snapshot_path = Settings.RESULTS_DIR / filename
        
        if not snapshot_path.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Snapshot not found"}
            )
        
        return FileResponse(snapshot_path, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error retrieving snapshot: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


# ============================================================================
# CAMERA MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/cameras")
async def get_cameras():
    """Get list of available CCTV cameras (video files)"""
    cameras = []
    if Settings.VIDEO_DIR.exists():
        for video_file in Settings.VIDEO_DIR.glob("*.mp4"):
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


@app.get("/api/cctv/config")
async def get_cctv_config():
    """Get saved CCTV camera configuration"""
    try:
        config_file = Path("cctv_config.json")
        
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
            
            return {
                "success": True,
                "cameras": config
            }
        
        return {"success": True, "cameras": []}
        
    except Exception as e:
        logger.error(f"Error reading CCTV config: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@app.post("/api/cctv/config")
async def save_cctv_config(cameras: Dict[str, Any]):
    """Save CCTV camera configuration"""
    try:
        config_file = Path("cctv_config.json")
        
        with open(config_file, "w") as f:
            json.dump(cameras, f, indent=2)
        
        logger.info(f"CCTV config saved: {len(cameras)} camera(s)")
        
        return {
            "success": True,
            "message": f"Configuration saved for {len(cameras)} camera(s)"
        }
    except Exception as e:
        logger.error(f"Error saving CCTV config: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


# ============================================================================
# STATIC FILES
# ============================================================================

# Mount frontend static files
try:
    app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
    logger.info("Frontend mounted successfully")
except Exception as e:
    logger.warning(f"Could not mount frontend: {e}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup - Create required directories"""
    # Create necessary directories
    Settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    Settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("DRISTI - Lost Person Detection System v2.0")
    logger.info("=" * 70)
    logger.info(f"Upload directory: {Settings.UPLOAD_DIR}")
    logger.info(f"Results directory: {Settings.RESULTS_DIR}")
    logger.info(f"Videos directory: {Settings.VIDEO_DIR}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("DRISTI system shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Settings.PORT)
