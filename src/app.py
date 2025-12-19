from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import os
import shutil
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import asyncio

from .config import settings
from .database import get_db, engine
from .models import User, TargetPerson, CCTV, SearchJob, Alert, Base
from .face_recognizer import face_recognizer
from . import schemas
from . import crud

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")

# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = crud.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Helper functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

# Routes
@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Authentication
@app.post(f"{settings.API_V1_PREFIX}/auth/register", response_model=schemas.User)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    return crud.create_user(db=db, user=user)

@app.post(f"{settings.API_V1_PREFIX}/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin
        }
    }

# Target Persons
@app.post(f"{settings.API_V1_PREFIX}/targets", response_model=schemas.TargetPerson)
async def create_target(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    photo: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Save uploaded photo
    photo_filename = f"target_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    photo_path = os.path.join(settings.UPLOAD_DIR, photo_filename)
    
    with open(photo_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
    
    # Process target person
    success, result = face_recognizer.set_target_person(photo_path, name)
    
    if not success:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to set target"))
    
    # Create target person in database
    target_data = schemas.TargetPersonCreate(
        name=name,
        description=description,
        photo_path=result["face_image_path"],
        embedding_path=result["embedding_path"]
    )
    
    return crud.create_target_person(db=db, target=target_data, user_id=current_user.id)

@app.get(f"{settings.API_V1_PREFIX}/targets", response_model=List[schemas.TargetPerson])
async def list_targets(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return crud.get_target_persons(db, user_id=current_user.id, skip=skip, limit=limit)

@app.get(f"{settings.API_V1_PREFIX}/targets/{{target_id}}", response_model=schemas.TargetPerson)
async def get_target(
    target_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    target = crud.get_target_person(db, target_id=target_id, user_id=current_user.id)
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return target

# CCTV Management
@app.post(f"{settings.API_V1_PREFIX}/cctvs", response_model=schemas.CCTV)
async def create_cctv(
    cctv: schemas.CCTVCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return crud.create_cctv(db=db, cctv=cctv, user_id=current_user.id)

@app.post(f"{settings.API_V1_PREFIX}/cctvs/{{cctv_id}}/upload")
async def upload_cctv_video(
    cctv_id: str,
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    cctv = crud.get_cctv(db, cctv_id=cctv_id, user_id=current_user.id)
    if not cctv:
        raise HTTPException(status_code=404, detail="CCTV not found")
    
    # Save video
    video_filename = f"cctv_{cctv_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = os.path.join(settings.UPLOAD_DIR, video_filename)
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Update CCTV with video path
    crud.update_cctv_video(db, cctv_id=cctv_id, video_path=video_path)
    
    return {"message": "Video uploaded successfully", "video_path": video_path}

# Search Jobs
@app.post(f"{settings.API_V1_PREFIX}/search", response_model=schemas.SearchJob)
async def create_search_job(
    search_data: schemas.SearchJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Create search job in database
    search_job = crud.create_search_job(db=db, search_data=search_data, user_id=current_user.id)
    
    # Start search in background
    background_tasks.add_task(
        process_search_job,
        search_job_id=search_job.id,
        db=db
    )
    
    return search_job

def process_search_job(search_job_id: str, db: Session):
    """Process search job in background"""
    try:
        # Get search job
        search_job = crud.get_search_job(db, search_job_id)
        if not search_job:
            return
        
        # Update status to processing
        crud.update_search_job_status(db, search_job_id, "processing", started_at=datetime.utcnow())
        
        # Get target embedding
        target = search_job.target
        if not target.embedding_path or not os.path.exists(target.embedding_path):
            raise Exception("Target embedding not found")
        
        target_embedding = np.load(target.embedding_path)
        
        # Get CCTV video
        cctv = search_job.cctv
        if not cctv.video_path or not os.path.exists(cctv.video_path):
            raise Exception("CCTV video not found")
        
        # Perform search
        result = face_recognizer.search_in_video(
            video_path=cctv.video_path,
            target_embedding=target_embedding,
            target_name=target.name,
            camera_name=cctv.name,
            frame_skip=search_job.frame_skip,
            similarity_threshold=search_job.similarity_threshold
        )
        
        if "error" in result:
            raise Exception(result["error"])
        
        # Update search job with results
        crud.update_search_job_results(
            db, 
            search_job_id,
            status="completed",
            matches_found=result["matches_found"],
            processing_time=result["processing_time"],
            output_video_path=result["output_video_path"],
            report_path=result["report_path"],
            match_details=result["match_details"],
            completed_at=datetime.utcnow()
        )
        
    except Exception as e:
        # Update search job as failed
        crud.update_search_job_status(
            db, 
            search_job_id, 
            "failed", 
            completed_at=datetime.utcnow()
        )
        print(f"Error processing search job {search_job_id}: {e}")

@app.get(f"{settings.API_V1_PREFIX}/search", response_model=List[schemas.SearchJob])
async def list_search_jobs(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return crud.get_search_jobs(db, user_id=current_user.id, skip=skip, limit=limit)

@app.get(f"{settings.API_V1_PREFIX}/search/{{search_id}}", response_model=schemas.SearchJob)
async def get_search_job(
    search_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    search_job = crud.get_search_job(db, search_id=search_id, user_id=current_user.id)
    if not search_job:
        raise HTTPException(status_code=404, detail="Search job not found")
    return search_job

# Results
@app.get(f"{settings.API_V1_PREFIX}/results/{{search_id}}/video")
async def get_result_video(
    search_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    search_job = crud.get_search_job(db, search_id=search_id, user_id=current_user.id)
    if not search_job or not search_job.output_video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        search_job.output_video_path,
        media_type="video/mp4",
        filename=os.path.basename(search_job.output_video_path)
    )

@app.get(f"{settings.API_V1_PREFIX}/results/{{search_id}}/report")
async def get_result_report(
    search_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    search_job = crud.get_search_job(db, search_id=search_id, user_id=current_user.id)
    if not search_job or not search_job.report_path:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        search_job.report_path,
        media_type="application/json",
        filename=os.path.basename(search_job.report_path)
    )

# System Info
@app.get(f"{settings.API_V1_PREFIX}/system/info")
async def system_info(current_user: User = Depends(get_current_active_user)):
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "face_recognition": {
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "detection_confidence": settings.DETECTION_CONFIDENCE,
            "frame_skip": settings.FRAME_SKIP
        },
        "storage": {
            "upload_dir": settings.UPLOAD_DIR,
            "results_dir": settings.RESULTS_DIR
        }
    }

# Run application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )