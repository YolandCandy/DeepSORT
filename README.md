# Car Tracking with DeepSORT and YOLO26

This repository contains a full pipeline for detecting and tracking cars using the **YOLO26** object detector and the **DeepSORT** tracking algorithm.

## Features
- **End-to-End Tracking**: Detects cars with YOLO26 and associates them across frames using DeepSORT.
- **Pre-trained Support**: Uses a pre-trained MobileNet feature extractor (Re-ID model) from `deep-sort-realtime` optimized for speed.
- **Training Support**: Includes a skeleton script `train_yolo.py` to fine-tune YOLO26 on the UA-DETRAC dataset.
- **Dual Output**: Displays the tracking in real-time and saves the result to `output.mp4`.

## Setup
1. **Activate the environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Download a test video** (if you don't have UA-DETRAC downloaded yet):
   ```powershell
   python download_test_video.py
   ```

## Usage

### Running the Tracker
To track cars on a video file, use the main `deepsort.py` script:
```powershell
python deepsort.py --video test_video.mp4 --model yolo/yolo26n.pt --save-video --show-video
```

**Available Arguments:**
- `--video`: Path to input video.
- `--output`: Path to save output video (default: `output.mp4`).
- `--trackeval-output`: Path to save TrackEval-formatted tracking results (default: `output.txt`).
- `--model`: Path to YOLO weights (default: `yolo/yolo26n.pt`).
- `--save-video`: Flag to save the processed output video.
- `--no-save-txt`: Flag to disable saving TrackEval results (results are saved by default).
- `--show-video`: Flag to show real-time tracking visualization.

### Training on UA-DETRAC
To train the YOLO26 detector on the UA-DETRAC dataset, ensure your data is in YOLO format with a `data.yaml` config file:
```powershell
python train_yolo.py --data path/to/ua_detrac/data.yaml --epochs 100
```
