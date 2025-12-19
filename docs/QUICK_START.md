# DRISTI Quick Start Guide

## ⚡ 60-Second Setup

### 1️⃣ Install Dependencies (1 min)
```bash
pip install -r requirements.txt
```

### 2️⃣ Verify System (30 seconds)
```bash
python test_system.py
```
You should see: `✓ All systems ready!`

### 3️⃣ Start Server (instant)
```bash
python main.py
```
You should see:
```
======================================================================
DRISTI - Lost Person Detection System
======================================================================
Starting server at http://localhost:8000
```

### 4️⃣ Open Browser (instant)
Navigate to: **http://localhost:8000**

## 📋 What You Should See

### Home Page
- Logo and title "DRISTI"
- Large upload area
- "Click to upload or drag and drop" text
- No login or authentication

### Upload Section
1. **Click** the upload area OR **Drag** a photo onto it
2. **Select** a JPG/PNG photo of a person
3. **Preview** appears below
4. **Click** "Search CCTV Footage" button

### Search Progress
- Spinner animation
- Message: "Searching CCTV footage..."
- Waits for backend to process

### Results
- Shows number of matches found
- Best match confidence percentage
- Camera names that detected the person
- Timestamp where person was found
- Thumbnail images from CCTV

## 📹 About the Demo Videos

The system includes 4 CCTV sample videos:
- **cctv1.mp4** (4.35 MB)
- **cctv2.mp4** (3.71 MB)  
- **cctv3.mp4** (2.76 MB)
- **cctv4.mp4** (2.47 MB)

These are crowded scene videos with multiple faces.

## 🎯 Try It Out

### Test with Sample Photo
1. Take a screenshot of a face from one of the videos
2. Upload it to the system
3. System will search all 4 videos
4. Should find matches from where that face appeared

### Or Use Your Own
1. Upload any clear face photo
2. System searches all CCTV videos
3. Returns matches if person appears in videos

## ⚙️ Adjust Sensitivity

Edit `search_service.py` if needed:

```python
# More matches but less accurate
self.similarity_threshold = 0.5

# Fewer matches but more accurate
self.similarity_threshold = 0.7

# Faster processing (skip more frames)
self.frame_skip = 10

# Slower but more thorough
self.frame_skip = 3
```

Then restart: `python main.py`

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No face detected" | Try a clearer photo, different angle |
| "Port 8000 in use" | `python main.py --port 8001` |
| Search taking too long | Increase `frame_skip` value |
| Not finding person | Decrease `similarity_threshold` |
| Server won't start | Run `python test_system.py` first |

## 📊 System Performance

| Task | Time |
|------|------|
| Upload photo | < 1 second |
| Extract face | < 1 second |
| Search 1 minute video | 5-10 seconds |
| Search all 4 videos (~12 min total) | 30-60 seconds |
| Display results | < 1 second |

## 🎓 How It Works (Simple Explanation)

1. **Upload Photo**
   - User provides photo of lost person
   - System detects face in photo
   - System converts face to unique "fingerprint" (embedding)

2. **Search Videos**
   - System reads each video frame by frame
   - Detects all faces in each frame
   - Converts each detected face to fingerprint
   - Compares with lost person's fingerprint

3. **Find Matches**
   - If fingerprints match (confidence > threshold)
   - System notes:
     - Which camera (video file)
     - What time in video (MM:SS)
     - How confident the match is (%)

4. **Show Results**
   - Frontend displays all matches
   - Shows thumbnail from video
   - Lists all details

## 📱 Using on Different Devices

### Same Machine
```
http://localhost:8000
```

### Different Computer on Network
```
http://<server-ip>:8000
```

Example: `http://192.168.1.100:8000`

### From Phone/Tablet
1. Find server's IP: `ipconfig` (Windows)
2. On phone, navigate to: `http://<IP>:8000`

## ✅ Verification Checklist

- [ ] Python installed and working
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] System test passes (`python test_system.py`)
- [ ] 4 CCTV videos present in CCTVS/ folder
- [ ] Server starts without errors
- [ ] Browser can access http://localhost:8000
- [ ] Upload area is visible
- [ ] Can select and preview a photo
- [ ] Search button works
- [ ] Results display correctly

## 🚀 Next Steps

1. **Test with real photos** - Try photos of different people
2. **Add more videos** - Replace CCTV videos with your own
3. **Adjust settings** - Fine-tune for your use case
4. **Deploy** - Put on a real server with proper IP

## 💡 Pro Tips

- **Better accuracy**: Use higher quality input photo
- **Faster search**: Use shorter videos or higher frame_skip
- **More matches**: Lower the similarity_threshold
- **Better lighting**: CCTV videos with good lighting = better results
- **Face clarity**: Clear, frontal faces detected better than side profiles

## 🆘 When Things Go Wrong

### Step 1: Check System Health
```bash
python test_system.py
```

### Step 2: Check Logs
```bash
type logs/face_recognition.log
```

### Step 3: Check Videos Exist
```bash
dir CCTVS
```

### Step 4: Restart Server
- Stop current process (Ctrl+C)
- Run again: `python main.py`

### Step 5: Ask for Help
If still stuck, include:
- Error message from server
- Output of `python test_system.py`
- What you were trying to do

---

**Ready to find missing persons? Let's get started! 🚀**
