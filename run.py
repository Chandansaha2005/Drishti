import os
import sys
import uvicorn

# Get port from environment or default to 8000
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    os.makedirs('CCTVS', exist_ok=True)
    
    # Run the app
    uvicorn.run(
        'src.backend.app:app',
        host=HOST,
        port=PORT,
        reload=False  # Disable reload in production
    )
