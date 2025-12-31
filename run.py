import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backend.app import app
import uvicorn

if __name__ == "__main__":
    print("Starting Drishti - Lost Person Detection System...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
