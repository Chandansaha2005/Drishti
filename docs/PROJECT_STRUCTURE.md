# Drishti - Lost Person Search & Face Recognition System

## Project Structure

```
drishti/
├── backend/                    # Python backend application
│   ├── __init__.py
│   ├── app.py                 # FastAPI application entry point
│   ├── main.py                # Main application logic
│   ├── config.py              # Configuration management
│   ├── database.py            # Database connection
│   ├── models.py              # SQLAlchemy models
│   ├── schemas.py             # Pydantic schemas
│   ├── crud.py                # CRUD operations
│   ├── face_recognizer.py     # Face recognition logic
│   └── search_service.py      # Search and matching service
│
├── frontend/                   # Frontend application
│   ├── pages/                 # HTML pages
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── search.html
│   │   ├── results.html
│   │   ├── targets.html
│   │   └── cctv.html
│   ├── js/                    # JavaScript files
│   │   ├── script.js
│   │   ├── api.js
│   │   └── auth.js
│   ├── css/                   # Stylesheets
│   │   └── style.css
│   └── assets/                # Images, fonts, etc.
│
├── tests/                      # Test suite
│   └── test_system.py
│
├── config/                     # Configuration files
│   └── config.py              # Environment configuration
│
├── data/                       # Data directory (git ignored)
│   ├── cctvs/                 # CCTV video files
│   ├── uploads/               # User uploads
│   └── results/               # Search results
│
├── docs/                       # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── SETUP_GUIDE.md
│   ├── QUICK_START.md
│   ├── PROJECT_SUMMARY.md
│   ├── INDEX.md
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── FINAL_REPORT.txt
│   └── STATUS_SUMMARY.txt
│
├── logs/                       # Application logs (git ignored)
│
├── models/                     # Pre-trained ML models (git ignored)
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── run.bat                    # Windows batch runner
```

## Quick Start

See [QUICK_START.md](docs/QUICK_START.md) for detailed setup instructions.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture details.

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit your changes: `git commit -am 'Add new feature'`
3. Push to the branch: `git push origin feature/your-feature`
4. Submit a Pull Request

## License

[Add your license here]
