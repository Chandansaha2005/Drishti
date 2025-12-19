from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    searches = relationship("SearchJob", back_populates="user")
    targets = relationship("TargetPerson", back_populates="user")

class TargetPerson(Base):
    __tablename__ = "target_persons"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    photo_path = Column(String(255), nullable=False)
    embedding_path = Column(String(255))  # Path to saved embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="targets")
    searches = relationship("SearchJob", back_populates="target")

class CCTV(Base):
    __tablename__ = "cctvs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    location = Column(String(200))
    description = Column(Text)
    video_path = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchJob(Base):
    __tablename__ = "search_jobs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    target_id = Column(String(36), ForeignKey("target_persons.id"), nullable=False)
    cctv_id = Column(String(36), ForeignKey("cctvs.id"), nullable=False)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    similarity_threshold = Column(Float, default=0.75)
    frame_skip = Column(Integer, default=3)
    
    # Results
    matches_found = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    output_video_path = Column(String(255))
    report_path = Column(String(255))
    
    # Match details (stored as JSON)
    match_details = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="searches")
    target = relationship("TargetPerson", back_populates="searches")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    search_job_id = Column(String(36), ForeignKey("search_jobs.id"), nullable=False)
    frame_number = Column(Integer)
    time_seconds = Column(Float)
    similarity = Column(Float)
    snapshot_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)