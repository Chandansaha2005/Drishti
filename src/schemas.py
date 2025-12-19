from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_admin: bool = False

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Target Person schemas
class TargetPersonBase(BaseModel):
    name: str
    description: Optional[str] = None

class TargetPersonCreate(TargetPersonBase):
    photo_path: str
    embedding_path: Optional[str] = None

class TargetPerson(TargetPersonBase):
    id: str
    user_id: int
    photo_path: str
    embedding_path: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

# CCTV schemas
class CCTVBase(BaseModel):
    name: str
    location: Optional[str] = None
    description: Optional[str] = None

class CCTVCreate(CCTVBase):
    pass

class CCTV(CCTVBase):
    id: str
    user_id: int
    video_path: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Search Job schemas
class SearchJobBase(BaseModel):
    target_id: str
    cctv_id: Optional[str] = None
    similarity_threshold: Optional[float] = 0.75
    frame_skip: Optional[int] = 3

class SearchJobCreate(SearchJobBase):
    pass

class SearchJob(SearchJobBase):
    id: str
    user_id: int
    status: str
    matches_found: Optional[int] = 0
    processing_time: Optional[float] = 0.0
    output_video_path: Optional[str] = None
    report_path: Optional[str] = None
    match_details: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Response schemas
class MessageResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str