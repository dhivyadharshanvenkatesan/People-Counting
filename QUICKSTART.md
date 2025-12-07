# QUICK START GUIDE - Pedestrian Tracking System

## 📌 5-Minute Setup

### Step 1: Create Project Folder
```bash
mkdir pedestrian_tracking_system
cd pedestrian_tracking_system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy scipy Pillow pandas matplotlib tqdm pyyaml torch torchvision
```

### Step 4: Create Code Files
Copy these Python files into your project:
- `centroid_tracker.py`
- `yolo_detector.py`
- `mot_dataset_processor.py`
- `run_video_tracking.py`
- `run_mot_dataset.py`

### Step 5: Verify Installation
```bash
python -c "from ultralytics import YOLO; print('✓ Ready to track!')"
```

---

## 🎥 USAGE: Process Your Video

### Option A: Simple Video Processing
```bash
python run_video_tracking.py your_video.mp4
```

This will:
1. Create `your_video_tracked.mp4` with tracking overlays
2. Display real-time tracking (press 'q' to quit)
3. Print pedestrian counts

### Option B: Save Output (without display)
```bash
python run_video_tracking.py your_video.mp4 --output tracked.mp4 --no-display
```

### Option C: Adjust Settings
```bash
python run_video_tracking.py your_video.mp4 --conf 0.6 --device cpu --output out.mp4
```

**Parameters:**
- `--conf`: Detection confidence (0.0-1.0, default 0.5)
- `--device`: 0 for GPU, 'cpu' for CPU
- `--output`: Output file path
- `--no-display`: Don't show video window

---

## 📊 USAGE: Process MOT17 Dataset

### Step 1: Download MOT17
1. Visit https://motchallenge.net/
2. Register (free)
3. Download MOT17 dataset
4. Extract to `data/MOT17/`

### Step 2: Process Single Sequence
```bash
python run_mot_dataset.py data/MOT17 --split train --sequence MOT17-02-DPM --output results/
```

### Step 3: Process All Sequences
```bash
python run_mot_dataset.py data/MOT17 --split train --output results/
```

### Step 4: View Results
Output videos saved in `results/` folder with tracking overlays.

---

## 🔍 UNDERSTANDING MOT17 STRUCTURE

Your downloaded MOT17 should look like:
```
data/MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/              ← Frame images (000001.jpg, 000002.jpg, ...)
│   │   ├── gt/
│   │   │   └── gt.txt         ← Ground truth annotations
│   │   ├── det/
│   │   │   └── det.txt        ← Detection file
│   │   └── seqinfo.ini        ← Sequence metadata
│   ├── MOT17-04-DPM/
│   └── ... more sequences
└── test/
    └── ... test sequences
```

**Key Files:**
- `img1/`: Contains frame images (JPG format)
- `gt.txt`: Format: `<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>`
- `seqinfo.ini`: Contains metadata (FPS, resolution, etc.)

---

## 📋 FILE PLACEMENT CHECKLIST

Make sure your project folder looks like this:

```
pedestrian_tracking_system/
├── venv/                          ✓ Virtual environment
├── data/                          
│   ├── MOT17/                     ✓ Dataset (if using MOT17)
│   ├── videos/                    ✓ Your video files
│   └── output/                    ✓ Output folder
├── centroid_tracker.py            ✓ Tracker module
├── yolo_detector.py               ✓ Detector module  
├── mot_dataset_processor.py       ✓ Dataset processor
├── run_video_tracking.py          ✓ Main video script
├── run_mot_dataset.py             ✓ MOT17 processor script
└── requirements.txt               ✓ Dependencies list
```

---

## 🚀 COMMON COMMANDS

### Run on your video
```bash
python run_video_tracking.py data/videos/my_video.mp4 --output data/output/tracked.mp4
```

### Run on MOT17 training set with GPU
```bash
python run_mot_dataset.py data/MOT17 --split train --device 0 --output data/output/
```

### Run single MOT17 sequence with CPU
```bash
python run_mot_dataset.py data/MOT17 --split train --sequence MOT17-02-DPM --device cpu
```

### Enable display for debugging
```bash
python run_video_tracking.py my_video.mp4 --conf 0.4
```
(Press 'q' to stop)

---

## 🎯 EXPECTED OUTPUT

When processing a video, you'll see:
```
Processing: my_video.mp4
Resolution: 1920x1080 @ 30 FPS
Total frames: 900

Progress: 30/900 (3.3%) | Up: 2 | Down: 5
Progress: 60/900 (6.7%) | Up: 5 | Down: 12
...

Processing complete!
Total frames processed: 900
People crossing UP: 45
People crossing DOWN: 58
Total people: 103
Output saved to: my_video_tracked.mp4
```

---

## ⚠️ TROUBLESHOOTING

### Issue: "Module not found"
```bash
# Ensure virtual environment is activated
# Should see (venv) at start of terminal prompt
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
```bash
# Use smaller model or CPU
python run_video_tracking.py my_video.mp4 --device cpu
```

### Issue: No detections / Low accuracy
```bash
# Lower confidence threshold
python run_video_tracking.py my_video.mp4 --conf 0.3
```

### Issue: Slow processing
```bash
# Use GPU and smaller model
python run_video_tracking.py my_video.mp4 --device 0
```

---

## 📝 NEXT STEPS

1. **Basic Testing**: Process a simple video with `run_video_tracking.py`
2. **MOT17 Exploration**: Download MOT17 and process one sequence
3. **Customization**: Modify tracking parameters in code
4. **Evaluation**: Compare results with ground truth in MOT17

---

## 📞 Need Help?

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **MOTChallenge**: https://motchallenge.net/
- **OpenCV Docs**: https://docs.opencv.org/

---

**You're all set!** 🎉 Start with:
```bash
python run_video_tracking.py your_video.mp4
```
