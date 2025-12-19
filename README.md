# Face Recognition Backend

Minimal FastAPI-based backend for face recognition tasks.

Structure:

- `app.py` - FastAPI application
- `face_recognizer.py` - Core recognition logic using `face_recognition` lib
- `database.py`, `models.py` - DB scaffolding (SQLAlchemy)
- `config.py` - Project settings
- `uploads/`, `results/`, `logs/` - directories for runtime files

Run:

1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `uvicorn app:app --reload`
