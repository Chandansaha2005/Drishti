from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from passlib.context import CryptContext
import models
import schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User CRUD
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_admin=user.is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

# Target Person CRUD
def create_target_person(db: Session, target: schemas.TargetPersonCreate, user_id: int):
    db_target = models.TargetPerson(
        user_id=user_id,
        name=target.name,
        description=target.description,
        photo_path=target.photo_path,
        embedding_path=target.embedding_path
    )
    db.add(db_target)
    db.commit()
    db.refresh(db_target)
    return db_target

def get_target_person(db: Session, target_id: str, user_id: int):
    return db.query(models.TargetPerson).filter(
        models.TargetPerson.id == target_id,
        models.TargetPerson.user_id == user_id
    ).first()

def get_target_persons(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.TargetPerson).filter(
        models.TargetPerson.user_id == user_id
    ).order_by(desc(models.TargetPerson.created_at)).offset(skip).limit(limit).all()

# CCTV CRUD
def create_cctv(db: Session, cctv: schemas.CCTVCreate, user_id: int):
    db_cctv = models.CCTV(
        user_id=user_id,
        name=cctv.name,
        location=cctv.location,
        description=cctv.description
    )
    db.add(db_cctv)
    db.commit()
    db.refresh(db_cctv)
    return db_cctv

def get_cctv(db: Session, cctv_id: str, user_id: int):
    return db.query(models.CCTV).filter(
        models.CCTV.id == cctv_id,
        models.CCTV.user_id == user_id
    ).first()

def get_cctvs(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.CCTV).filter(
        models.CCTV.user_id == user_id
    ).order_by(desc(models.CCTV.created_at)).offset(skip).limit(limit).all()

def update_cctv_video(db: Session, cctv_id: str, video_path: str):
    cctv = db.query(models.CCTV).filter(models.CCTV.id == cctv_id).first()
    if cctv:
        cctv.video_path = video_path
        db.commit()
        db.refresh(cctv)
    return cctv

# Search Job CRUD
def create_search_job(db: Session, search_data: schemas.SearchJobCreate, user_id: int):
    db_search = models.SearchJob(
        user_id=user_id,
        target_id=search_data.target_id,
        cctv_id=search_data.cctv_id,
        similarity_threshold=search_data.similarity_threshold,
        frame_skip=search_data.frame_skip
    )
    db.add(db_search)
    db.commit()
    db.refresh(db_search)
    return db_search

def get_search_job(db: Session, search_id: str, user_id: Optional[int] = None):
    query = db.query(models.SearchJob).filter(models.SearchJob.id == search_id)
    if user_id:
        query = query.filter(models.SearchJob.user_id == user_id)
    return query.first()

def get_search_jobs(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.SearchJob).filter(
        models.SearchJob.user_id == user_id
    ).order_by(desc(models.SearchJob.created_at)).offset(skip).limit(limit).all()

def update_search_job_status(db: Session, search_id: str, status: str, 
                           started_at=None, completed_at=None):
    search_job = db.query(models.SearchJob).filter(models.SearchJob.id == search_id).first()
    if search_job:
        search_job.status = status
        if started_at:
            search_job.started_at = started_at
        if completed_at:
            search_job.completed_at = completed_at
        db.commit()
        db.refresh(search_job)
    return search_job

def update_search_job_results(db: Session, search_id: str, status: str,
                            matches_found: int, processing_time: float,
                            output_video_path: str, report_path: str,
                            match_details: dict, completed_at=None):
    search_job = db.query(models.SearchJob).filter(models.SearchJob.id == search_id).first()
    if search_job:
        search_job.status = status
        search_job.matches_found = matches_found
        search_job.processing_time = processing_time
        search_job.output_video_path = output_video_path
        search_job.report_path = report_path
        search_job.match_details = match_details
        if completed_at:
            search_job.completed_at = completed_at
        db.commit()
        db.refresh(search_job)
    return search_job