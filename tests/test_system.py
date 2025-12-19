#!/usr/bin/env python
"""
DRISTI Test Script - Verify system components
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Package Imports")
    
    packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('torchvision', 'Torchvision'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name:20} - OK")
        except ImportError as e:
            print(f"✗ {name:20} - MISSING")
            all_ok = False
    
    return all_ok

def test_directories():
    """Check if necessary directories exist or can be created"""
    print_header("Testing Directory Structure")
    
    dirs = [
        Path('uploads'),
        Path('results'),
        Path('CCTVS'),
        Path('logs'),
    ]
    
    all_ok = True
    for dir_path in dirs:
        if dir_path.exists():
            print(f"✓ {str(dir_path):20} - EXISTS")
        else:
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"✓ {str(dir_path):20} - CREATED")
            except Exception as e:
                print(f"✗ {str(dir_path):20} - ERROR: {e}")
                all_ok = False
    
    return all_ok

def test_files():
    """Check if main files exist"""
    print_header("Testing Required Files")
    
    files = [
        Path('main.py'),
        Path('search_service.py'),
        Path('requirements.txt'),
        Path('Frontend/index.html'),
    ]
    
    all_ok = True
    for file_path in files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {str(file_path):30} - EXISTS ({size} bytes)")
        else:
            print(f"✗ {str(file_path):30} - MISSING")
            all_ok = False
    
    return all_ok

def test_videos():
    """Check for video files"""
    print_header("Testing Video Files")
    
    cctvs_dir = Path('CCTVS')
    if not cctvs_dir.exists():
        print("✗ CCTVS directory not found")
        return False
    
    videos = list(cctvs_dir.glob('*.mp4'))
    
    if not videos:
        print("⚠ No MP4 videos found in CCTVS/")
        print("  Please add your video files to the CCTVS directory")
        return False
    
    print(f"✓ Found {len(videos)} video file(s):")
    for video in videos:
        size_mb = video.stat().st_size / (1024*1024)
        print(f"  - {video.name:30} ({size_mb:.2f} MB)")
    
    return True

def test_syntax():
    """Test Python syntax"""
    print_header("Testing Python Syntax")
    
    import py_compile
    files = ['main.py', 'search_service.py']
    
    all_ok = True
    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"✓ {file:30} - SYNTAX OK")
        except py_compile.PyCompileError as e:
            print(f"✗ {file:30} - SYNTAX ERROR")
            print(f"  {e}")
            all_ok = False
    
    return all_ok

def test_models():
    """Test if models can be loaded"""
    print_header("Testing Model Loading")
    
    try:
        import torch
        import torchvision.models as models
        
        print("Loading ResNet18...")
        model = models.resnet18(pretrained=True)
        print("✓ ResNet18 model loaded successfully")
        
        import torch.nn as nn
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        print("✓ Model preprocessing successful")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "DRISTI System Verification Test" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        "Imports": test_imports(),
        "Directories": test_directories(),
        "Files": test_files(),
        "Videos": test_videos(),
        "Syntax": test_syntax(),
        "Models": test_models(),
    }
    
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} - {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All systems ready! You can now run: python main.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
