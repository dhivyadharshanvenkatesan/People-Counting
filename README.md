# Real-Time Pedestrian Tracking and Counting System

A production-ready Python implementation of real-time multi-object pedestrian tracking with dynamic polygon-based region-of-interest (ROI) selection. This system combines YOLOv8 for state-of-the-art object detection with efficient centroid-based tracking and comprehensive MOT17 benchmark evaluation.

## 🎯 Features

### Core Capabilities
- **Real-Time Detection**: YOLOv8 pedestrian detection at 7-11 FPS (CPU) / 30-60 FPS (GPU)
- **Multi-Object Tracking**: Centroid-based tracking with ID consistency across frames
- **Polygon ROI Counting**: Interactive 4-point polygon selection for flexible area monitoring
- **Directional Counting**: Entry/exit detection with state-machine-based crossing prevention
- **MOT17 Evaluation**: Comprehensive evaluation using standard MOT metrics (MOTA, Precision, Recall)

### Key Innovations
- ✨ **Interactive ROI Selection**: Define custom monitoring areas on-the-fly instead of fixed horizontal lines
- 🔍 **Smart Crossing Detection**: State machine prevents double-counting of same pedestrian
- 📊 **Full MOT Evaluation Framework**: Standard benchmark metrics for research-grade validation
- 🚀 **Production Ready**: Modular design, comprehensive error handling, extensive documentation

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🏗️ System Architecture

### Pipeline Overview

```
Video Input
    ↓
Frame Extraction (H×W×3)
    ↓
YOLO v8 Detection → Bounding Boxes [x1, y1, x2, y2, conf, cls]
    ↓
Centroid Tracking → Object IDs {id: (cx, cy)}
    ↓
Polygon Counter → Directional Counts (IN/OUT)
    ↓
Output → Visualization + Statistics
```

### Core Modules

| Module | Purpose | Key Algorithm |
|--------|---------|----------------|
| **yolo_detector.py** | Person detection per frame | YOLOv8 CNN |
| **centroid_tracker.py** | Temporal object association | Euclidean distance matching |
| **polygon_counter.py** | ROI crossing detection | Ray-casting point-in-polygon |
| **mot_evaluator.py** | Benchmark evaluation | IoU-based matching + MOTA calculation |
| **mot_dataset_processor.py** | MOT17 dataset loading | Sequence parser |

### Key Algorithms

**1. Centroid Matching** (Distance Formula)
```
d = sqrt((cx_det - cx_obj)² + (cy_det - cy_obj)²)
Match if: d < max_distance
```

**2. Polygon Crossing Detection** (Ray Casting)
```
point_inside = pointPolygonTest(polygon, point)
state_changed = (prev_side ≠ curr_side) AND NOT crossed_recently
```

**3. MOTA Evaluation** (Standard MOT Metric)
```
MOTA = 1 - (FP + FN + ID_Switches) / Total_GT_Detections
```

## 📦 Requirements

### System Requirements
- Python 3.8+
- RAM: 8GB minimum
- GPU: CUDA 11.8+ (optional, for acceleration)

### Python Dependencies

```
opencv-python==4.8.0
ultralytics==8.0.0          # YOLOv8
numpy==1.24.0
scipy==1.11.0               # Distance calculations
matplotlib==3.7.0           # Visualization (optional)
```

## 🚀 Installation

### Step 1: Clone or Setup Project

```bash
mkdir -p ~/pedestrian_tracking_system
cd ~/pedestrian_tracking_system
# If cloning from repo:
# git clone <repo_url> .
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install opencv-python==4.8.0
pip install ultralytics==8.0.0
pip install numpy==1.24.0
pip install scipy==1.11.0
```

### Step 4: Verify Installation

```bash
python -c "import cv2; import ultralytics; print('✓ Installation successful')"
```

### Step 5: Download YOLO Model (First Run)

The YOLOv8 model will auto-download on first use. Alternatively:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

## ⚡ Quick Start

### Real-Time Video Tracking (Recommended for Most Users)

```bash
# Basic usage
python scripts/run_video_tracking_polygon.py data/videos/sample.mp4

# With output name
python scripts/run_video_tracking_polygon.py video.mp4 --output result.mp4

# With custom confidence threshold
python scripts/run_video_tracking_polygon.py video.mp4 --conf 0.3

# Use GPU for faster processing
python scripts/run_video_tracking_polygon.py video.mp4 --device 0

# No display (background processing)
python scripts/run_video_tracking_polygon.py video.mp4 --no-display
```

### MOT17 Benchmark Evaluation

```bash
# Single sequence evaluation
python scripts/run_mot_dataset_with_eval.py --sequence MOT17-02-DPM --conf 0.3

# Full dataset evaluation
python scripts/run_mot_dataset_with_eval.py --conf 0.3 --device 0

# Dataset statistics without evaluation
python scripts/run_mot_dataset_with_eval.py --no-eval
```

## 📖 Usage Guide

### Real-Time Video Tracking

#### Interactive Polygon Selection

1. **Run the script**:
   ```bash
   python scripts/run_video_tracking_polygon.py video.mp4
   ```

2. **Select ROI on first frame**:
   - Click 4 points to define polygon
   - Points appear numbered (1, 2, 3, 4)
   - Press 'c' to confirm
   - Press 'r' to reset and reselect
   - Press 'ESC' to cancel

3. **System processes video**:
   - Detects and tracks pedestrians
   - Counts entries/exits through polygon
   - Shows real-time visualization
   - Displays IN/OUT counts

4. **Output generated**:
   - Video saved to `data/output/`
   - Statistics printed to console

#### Example Session

```
===============================================
POLYGON ROI TRACKING SYSTEM
===============================================

Video: mall_video.mp4
Resolution: 1920x1080 @ 30 FPS
Total frames: 1200

STEP 1: Select Polygon ROI
Instructions:
  1. Click 4 points to define the polygon
  2. Press 'c' to confirm selection
  3. Press 'r' to reset
  4. Press ESC to cancel

Polygon ROI selected with 4 points

STEP 2: Processing Video
Progress: 300/1200 (25.0%) | Detections: 8 | Tracked: 7 | IN: 5 | OUT: 3
Progress: 600/1200 (50.0%) | Detections: 12 | Tracked: 11 | IN: 23 | OUT: 19

===============================================
RESULTS
===============================================
Total frames processed: 1200
Total pedestrians detected: 87
Objects entering polygon: 134
Objects leaving polygon: 128
Total crossings: 262

✓ Output saved to: data/output/mall_video_polygon_tracked.mp4
===============================================
```

### MOT17 Benchmark Evaluation

#### Understanding Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MOTA** | 1 - (FP+FN+IDSW)/GT | Overall tracking accuracy (%) |
| **Precision** | TP/(TP+FP) | Detection accuracy (%) |
| **Recall** | TP/(TP+FN) | Detection coverage (%) |
| **ID Switches** | Count of ID changes | Tracking consistency |
| **FP** | False Positives | Incorrectly detected |
| **FN** | False Negatives | Missed detections |

#### Example Output

```
======================================================================
Evaluation Metrics for MOT17-02-DPM
======================================================================

Overall Performance:
  MOTA (Multi-Object Tracking Accuracy):    5.91%
  Precision:                               56.28%
  Recall:                                  33.74%
  F1 Score:                                42.31%

Trajectory Statistics:
  Mostly Tracked (MT):                     8 (12.9%)
  Partially Tracked (PT):                 15 (24.2%)
  Mostly Lost (ML):                       39 (62.9%)

Error Statistics:
  False Positives (FP):                    4870
  False Negatives (FN/Misses):            12312
  ID Switches:                              301

Dataset Statistics:
  Ground Truth Trajectories:                 62
  Predicted Trajectories:                    53
  Total Frames:                             600
  Total GT Detections:                    18581
  Total Predicted Detections:              6007
======================================================================
```

## ⚙️ Configuration

### YOLO Detection Parameters

Edit these in code or use command-line arguments:

```python
# Confidence threshold (0.0-1.0)
# Lower = more detections, more false positives
# Higher = fewer detections, higher precision
conf = 0.3  # Recommended for crowded scenes

# Model size
# n=nano (fastest), m=medium (balanced), x=xlarge (slowest)
model = "yolov8m.pt"

# Device: 0 for GPU, "cpu" for CPU
device = "cpu"
```

**Recommended Settings**:

| Scenario | Confidence | Model | Device |
|----------|-----------|-------|--------|
| Real-time (CPU) | 0.35 | yolov8m | cpu |
| Real-time (GPU) | 0.30 | yolov8l | 0 |
| High accuracy | 0.25 | yolov8x | 0 |
| Fast processing | 0.50 | yolov8n | 0 |

### Tracker Parameters

```python
max_disappeared = 50        # Frames before deregistering (1-2 sec @ 30fps)
max_distance = 50           # Max pixel distance for association
buffer_distance = 30        # Distance for crossing reset
```

### MOT Evaluation Parameters

```python
iou_threshold = 0.5        # Intersection over Union threshold for matching
mt_threshold = 0.8         # Mostly Tracked (coverage ≥ 80%)
ml_threshold = 0.2         # Mostly Lost (coverage ≤ 20%)
```

## 📊 Results

### Experimental Results on MOT17

**Dataset**: MOT17 (21 sequences, 11,000+ frames, HD resolution 1920×1080)

**Counting Results**:
```
Total Sequences Processed: 20 (7 unique × 3 detectors)
Total Pedestrians Counted: 645
Average per sequence: 32.3 pedestrians
```

**Evaluation Metrics**:
```
Average MOTA:           -4.97%
Average Precision:      52.05%
Average Recall:         50.67%
Total False Positives:  116,886
Total False Negatives:  178,833
Total ID Switches:      6,108
```

**Per-Sequence Breakdown**:

| Sequence | MOTA | Precision | Recall | Count |
|----------|------|-----------|--------|-------|
| MOT17-02 | 5.91% | 56.28% | 33.74% | 18 |
| MOT17-04 | 33.96% | 78.92% | 46.68% | 13 |
| MOT17-05 | -34.55% | 40.42% | 59.22% | 71 |
| MOT17-09 | -20.23% | 44.61% | 66.70% | 26 |
| MOT17-10 | 0.44% | 52.15% | 51.71% | 9 |
| MOT17-11 | -9.67% | 46.60% | 59.99% | 61 |
| MOT17-13 | -10.63% | 45.40% | 36.67% | 17 |

### Performance Benchmarks

**Processing Speed**:

| Component | CPU (ms) | GPU (ms) |
|-----------|----------|----------|
| YOLO Detection | 80-120 | 15-25 |
| Centroid Tracking | 2-5 | N/A |
| Polygon Counting | 1-3 | N/A |
| **Total per Frame** | **85-128** | **17-33** |
| **FPS (Real-time)** | **7-11** | **30-60** |

## 📁 Project Structure

```
pedestrian_tracking_system/
│
├── scripts/                          # All Python source code
│   ├── yolo_detector.py             # YOLOv8 detection engine
│   ├── centroid_tracker.py          # Centroid-based tracking
│   ├── polygon_counter.py           # Polygon ROI counting
│   ├── mot_evaluator.py             # MOT evaluation metrics
│   ├── mot_dataset_processor.py     # MOT17 dataset loading
│   ├── run_video_tracking_polygon.py    # Main: Video tracking
│   └── run_mot_dataset_with_eval.py     # Main: MOT17 evaluation
│
├── data/
│   ├── MOT17/                       # MOT17 dataset (download separately)
│   │   ├── train/                   # Training sequences
│   │   │   ├── MOT17-02-DPM/
│   │   │   ├── MOT17-04-FRCNN/
│   │   │   └── ...
│   │   └── test/                    # Test sequences
│   ├── videos/                      # Input video files
│   │   └── sample.mp4
│   └── output/                      # Generated results
│       ├── tracked_videos/
│       └── debug/
│
├── venv/                            # Virtual environment (auto-created)
│
├── yolov8m.pt                       # YOLO model weights
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── PEDESTRIAN_TRACKING_REPORT.pdf   # Detailed technical report
```

### Directory Descriptions

- **scripts/**: All Python implementation files
  - Modules for detection, tracking, counting
  - Main execution scripts
  - Optional diagnostic tools

- **data/MOT17/**: MOT17 benchmark dataset
  - Must be downloaded separately
  - Structure: `train/sequence_name/img1/` + `gt/gt.txt`

- **data/videos/**: Input video files for processing
  - Supported formats: MP4, AVI, MOV, etc.

- **data/output/**: Generated results
  - Output videos with visualizations
  - Statistics and metrics
  - Debug images

## 🧠 Algorithms

### 1. YOLOv8 Detection

**Purpose**: Frame-by-frame pedestrian detection

**Key Parameters**:
- Confidence threshold: 0.3 (crowded scenes)
- Input resolution: 1920×1080
- Output: Bounding boxes [x1, y1, x2, y2, confidence, class]

**Formula**:
```
detection_kept if: confidence ≥ threshold
```

### 2. Centroid Tracking

**Purpose**: Maintain object IDs across frames

**Algorithm Steps**:
1. Compute centroid: (cx, cy) = ((x1+x2)/2, (y1+y2)/2)
2. Build distance matrix: D[i,j] = euclidean_distance(obj_i, det_j)
3. Greedy matching: Sort by distance, match lowest first
4. Update: Matched objects get new positions
5. Deregister: Objects missing > max_disappeared frames

**Distance Metric**:
```
d = sqrt((cx_det - cx_obj)² + (cy_det - cy_obj)²)
```

### 3. Polygon Crossing Detection

**Purpose**: Count entry/exit through flexible ROI

**Algorithm**:
1. User selects 4 points → defines polygon
2. For each tracked object:
   - Compute point-in-polygon using ray casting
   - Track state: inside/outside
   - Detect crossing: state change
   - Count: increment IN/OUT counter

**State Machine**:
```
if (prev_state ≠ curr_state) AND NOT recently_crossed:
    if curr_state == "inside":
        count_in += 1
    else:
        count_out += 1
    crossed = True

if distance_to_boundary > buffer_distance:
    crossed = False  # Reset for next crossing
```

### 4. MOT Evaluation

**Purpose**: Standard benchmark evaluation

**Metrics**:

**MOTA** (Multi-Object Tracking Accuracy):
```
MOTA = 1 - (FP + FN + ID_Switches) / Total_GT_Detections
```

**Precision & Recall**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**IoU Matching**:
```
IoU = Intersection_Area / Union_Area
Match if: IoU ≥ 0.5
```

## 🐛 Troubleshooting

### Issue: "No module named 'ultralytics'"

**Solution**:
```bash
pip install ultralytics
# Or reinstall requirements
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Use CPU instead of GPU
python scripts/run_video_tracking_polygon.py video.mp4 --device cpu

# Or use smaller YOLO model
# In code: model = "yolov8n.pt"  # nano instead of medium
```

### Issue: "MOT17 dataset not found"

**Solution**:
```bash
# Download MOT17 from: https://motchallenge.net/
# Extract to data/MOT17/

# Expected structure:
data/MOT17/
  ├── train/
  │   ├── MOT17-02-DPM/
  │   │   ├── img1/
  │   │   ├── gt/
  │   │   └── seqinfo.ini
  │   └── ...
  └── test/
      └── ...
```

### Issue: Polygon selection window not visible

**Solution**:
```bash
# Window might be behind other windows
# Press Alt+Tab to find it (Windows)
# Bring window to foreground

# Or run with explicit display:
python scripts/run_video_tracking_polygon.py video.mp4 --no-display
```

### Issue: Very low detection on crowded scenes

**Solution**:
```bash
# Lower confidence threshold for more detections
python scripts/run_video_tracking_polygon.py video.mp4 --conf 0.3

# Or use larger YOLO model
# In code: model = "yolov8l.pt" or "yolov8x.pt"
```

### Issue: High ID switches / Tracking inconsistency

**Solution**:
```bash
# Adjust tracker parameters in centroid_tracker.py
max_distance = 100  # Increase from 50
max_disappeared = 100  # Increase from 50
```

## 📚 References

### Academic Papers
1. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016
2. Ultralytics, "YOLOv8: State-of-the-art Detection", 2023
3. Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric", ICIP 2017
4. Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv:1603.00831

### Datasets
- **MOT17**: https://motchallenge.net/
- **MOT Challenge**: Multi-Object Tracking Benchmark

### Tools & Libraries
- **OpenCV**: https://opencv.org/
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **NumPy**: https://numpy.org/
- **SciPy**: https://www.scipy.org/

## 📄 Citation

If you use this system in your research, please cite:

```bibtex
@software{pedestrian_tracking_2025,
  title={Real-Time Pedestrian Tracking and Counting System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pedestrian_tracking_system}
}
```

## 📝 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add Kalman filtering for motion prediction
- [ ] Integrate Deep SORT appearance features
- [ ] Multi-camera tracking support
- [ ] GPU acceleration optimization
- [ ] Additional benchmark datasets
- [ ] Web interface for real-time monitoring

## 📞 Support

For issues, questions, or suggestions:

1. Check **Troubleshooting** section above
2. Review **PEDESTRIAN_TRACKING_REPORT.pdf** for detailed documentation
3. Check script comments and docstrings
4. Create an issue with detailed description

## 🎯 Future Enhancements

**Short-term**:
- [ ] Kalman filter integration
- [ ] Parameter auto-tuning
- [ ] Appearance feature extraction
- [ ] Multi-threaded processing

**Long-term**:
- [ ] Multi-camera tracking
- [ ] Cloud deployment
- [ ] Mobile app integration
- [ ] Real-time dashboard

---

## 📖 Documentation Files

This project includes comprehensive documentation:

- **README.md** (this file) - Quick start and overview
- **PEDESTRIAN_TRACKING_REPORT.pdf** - Detailed technical report (30+ pages)
  - Problem background and motivation
  - System architecture and algorithms
  - Complete implementation details
  - Experimental results and analysis
  - Conclusions and future work
- **COMPLETE-FILES-DOCUMENTATION.md** - Module-by-module breakdown
- **SCREENSHOT_GUIDELINES.md** - Visual setup instructions
- **FIX-EVALUATION-QUICK-GUIDE.md** - Configuration troubleshooting

---

**Happy Tracking! 🎯**

For questions or improvements, refer to the comprehensive technical report included in the project package.

---

*Last Updated*: November 2025
*Version*: 1.0
*Status*: Production Ready
