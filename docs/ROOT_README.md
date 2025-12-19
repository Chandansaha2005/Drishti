# Drishti - Lost Person Face Recognition & Search System

> A comprehensive AI-powered system for searching lost persons using facial recognition technology across CCTV networks.

## 📋 Project Overview

Drishti is an intelligent surveillance system that uses advanced face recognition algorithms to help locate missing persons by analyzing CCTV footage from multiple sources. The system combines deep learning with real-time video processing to provide quick and accurate results.

## 🗂️ New Project Structure

We've reorganized the project for better maintainability and GitHub publication:

```
drishti/
├── backend/              # FastAPI backend application
├── frontend/             # Web UI (HTML/JS/CSS)
├── tests/               # Test suite
├── config/              # Configuration files
├── data/                # Data directory (uploads, results, CCTV)
├── docs/                # Documentation
├── logs/                # Application logs
├── models/              # Pre-trained ML models
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── run.bat             # Run script
```

## 📋 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed folder structure explanation
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guide to migrate files to new structure
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Quick start guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Detailed setup instructions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drishti.git
   cd drishti
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   # On Windows
   run.bat
   
   # On Linux/Mac
   python backend/main.py
   ```

6. **Access the application**
   - Open browser: `http://localhost:8000`

## 🏗️ Architecture

### Backend Stack
- **Framework**: FastAPI
- **Database**: SQLite / PostgreSQL
- **Face Recognition**: face_recognition library
- **Video Processing**: OpenCV

### Frontend Stack
- **HTML5/CSS3**
- **Vanilla JavaScript**
- **REST API Integration**

## 🔧 Configuration

Create a `.env` file based on `.env.example`:

```env
DATABASE_URL=sqlite:///./drishti.db
API_PORT=8000
API_HOST=0.0.0.0
DEBUG=False
FACE_RECOGNITION_MODEL=hog
FACE_MATCH_TOLERANCE=0.6
```

## 📁 File Organization Guide

### After Migration

Move your existing files to the new structure using the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

**Key Points:**
- All Python backend files go to `backend/`
- Frontend files go to `frontend/` with subdirectories
- Documentation goes to `docs/`
- Data files go to `data/` (git ignored)
- Tests go to `tests/`

## 🔐 Git Setup

Already created:
- ✅ `.gitignore` - Comprehensive ignore rules
- ✅ `.env.example` - Environment template

### Initialize Repository

```bash
git init
git add .
git commit -m "Initial commit: Drishti face recognition system"
git remote add origin https://github.com/yourusername/drishti.git
git branch -M main
git push -u origin main
```

## 📊 Features

- ✨ Real-time face detection and recognition
- 📹 Multi-camera CCTV integration
- 🔍 Advanced search capabilities
- 📊 Search results visualization
- 🗄️ Persistent data storage
- 🎨 Responsive web interface
- 📈 Performance analytics

## 🧪 Testing

```bash
python -m pytest tests/
```

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Commit changes: `git commit -m 'Add amazing feature'`
3. Push to branch: `git push origin feature/amazing-feature`
4. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 📞 Support

For issues and questions:
1. Check [docs/QUICK_START.md](docs/QUICK_START.md)
2. Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. Open an issue on GitHub

## 🙏 Acknowledgments

- Special thanks to face_recognition library developers
- OpenCV community
- FastAPI framework team

---

**Status**: Ready for GitHub publication ✅
