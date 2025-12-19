# File Migration Guide

## Overview
This guide explains how to reorganize your project files according to the new structure for GitHub publication.

## Steps to Reorganize

### 1. Move Backend Python Files
Move these files from root to `backend/`:
- `app.py` → `backend/app.py`
- `main.py` → `backend/main.py`
- `config.py` → `backend/config.py`
- `database.py` → `backend/database.py`
- `models.py` → `backend/models.py`
- `schemas.py` → `backend/schemas.py`
- `crud.py` → `backend/crud.py`
- `face_recognizer.py` → `backend/face_recognizer.py`
- `search_service.py` → `backend/search_service.py`

### 2. Reorganize Frontend Files
Move these files from `Frontend/` to `frontend/`:
- `index.html`, `dashboard.html`, `search.html`, `results.html`, `targets.html`, `cctv.html` → `frontend/pages/`
- `script.js`, `api.js`, `auth.js` → `frontend/js/`
- `style.css` → `frontend/css/`
- `assets/*` → `frontend/assets/` (already has .gitkeep)

### 3. Move Documentation Files
Move these files to `docs/`:
- `README.md` → `docs/README.md`
- `ARCHITECTURE.md` → `docs/ARCHITECTURE.md`
- `SETUP_GUIDE.md` → `docs/SETUP_GUIDE.md`
- `QUICK_START.md` → `docs/QUICK_START.md`
- `PROJECT_SUMMARY.md` → `docs/PROJECT_SUMMARY.md`
- `INDEX.md` → `docs/INDEX.md`
- `IMPLEMENTATION_COMPLETE.md` → `docs/IMPLEMENTATION_COMPLETE.md`
- `FINAL_REPORT.txt` → `docs/FINAL_REPORT.txt`
- `STATUS_SUMMARY.txt` → `docs/STATUS_SUMMARY.txt`

### 4. Move Test Files
Move `test_system.py` → `tests/test_system.py`

### 5. Move Data Files
- CCTV files from `CCTVS/` → `data/cctvs/`
- Uploads from `uploads/` → `data/uploads/`
- Results from `results/` → `data/results/`

### 6. Update Import Paths
After moving Python files to `backend/`, update imports in your code:

**Example changes:**
```python
# Old
from models import User
from database import SessionLocal

# New
from backend.models import User
from backend.database import SessionLocal
```

Or if running from root:
```python
import sys
sys.path.insert(0, 'backend')
from models import User
```

### 7. Update Frontend Import Paths
Update HTML script references:
```html
<!-- Old -->
<script src="script.js"></script>
<script src="api.js"></script>

<!-- New -->
<script src="js/script.js"></script>
<script src="js/api.js"></script>
```

### 8. Files to Keep in Root
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `.env.example` - Environment template
- `run.bat` - Run script
- `PROJECT_STRUCTURE.md` - This structure guide

## Automated Migration Script (PowerShell)

```powershell
# Move backend files
Move-Item -Path ".\app.py" -Destination ".\backend\" -Force
Move-Item -Path ".\main.py" -Destination ".\backend\" -Force
Move-Item -Path ".\config.py" -Destination ".\backend\" -Force
Move-Item -Path ".\database.py" -Destination ".\backend\" -Force
Move-Item -Path ".\models.py" -Destination ".\backend\" -Force
Move-Item -Path ".\schemas.py" -Destination ".\backend\" -Force
Move-Item -Path ".\crud.py" -Destination ".\backend\" -Force
Move-Item -Path ".\face_recognizer.py" -Destination ".\backend\" -Force
Move-Item -Path ".\search_service.py" -Destination ".\backend\" -Force

# Move frontend files
Move-Item -Path ".\Frontend\*.html" -Destination ".\frontend\pages\" -Force
Move-Item -Path ".\Frontend\*.js" -Destination ".\frontend\js\" -Force
Move-Item -Path ".\Frontend\*.css" -Destination ".\frontend\css\" -Force

# Move docs
Move-Item -Path ".\*.md" -Destination ".\docs\" -Force
Move-Item -Path ".\*.txt" -Destination ".\docs\" -Force

# Move test
Move-Item -Path ".\test_system.py" -Destination ".\tests\" -Force

# Move data
Move-Item -Path ".\CCTVS\*" -Destination ".\data\cctvs\" -Force
Move-Item -Path ".\uploads\*" -Destination ".\data\uploads\" -Force
Move-Item -Path ".\results\*" -Destination ".\data\results\" -Force
```

## After Migration

1. **Update `requirements.txt`** to ensure all dependencies are correct
2. **Test the application** to ensure all imports work correctly
3. **Update CI/CD configurations** if any
4. **Commit to Git** with a message like: "refactor: reorganize project structure for GitHub publication"

## Files to Delete After Migration

Once everything is moved:
- Delete empty `Frontend/` folder
- Delete empty `CCTVS/` folder (if empty)
- Delete empty `uploads/` folder (if empty)
- Delete empty `results/` folder (if empty)
- Delete `sample_lost.jpg` and other test files in root if not needed

## Next Steps

After organizing:
1. Create a `.gitignore` file (already created in root)
2. Initialize Git: `git init`
3. Add files: `git add .`
4. Initial commit: `git commit -m "Initial commit: Drishti face recognition system"`
5. Add remote: `git remote add origin <your-repo-url>`
6. Push: `git push -u origin main`
