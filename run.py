import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backend.app import app
import uvicorn

if __name__ == "__main__":
    print("Starting Drishti - Lost Person Detection System...")
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
