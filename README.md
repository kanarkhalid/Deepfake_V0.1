# Deepfake Face Swap App

A real-time face-swapping application that loads a source photo and swaps it onto your face using your webcam feed. It uses **InsightFace** for deep learning-based face swapping and **GFPGAN** for high-quality face enhancement (restoring details and sharpness).

---

## Features

- **Real-Time Swap:** High-performance face detection and replacement directly on a webcam stream.
- **HQ Enhancement Mode:** Uses GFPGAN to enhance the resolution (512x512) and restore facial details (pores, eyes, hair) of the swapped face.
- **GPU Accelerated:** Automatically detects and utilizes NVIDIA CUDA via ONNX Runtime for smooth, high-FPS swapping.
- **Minimal HUD Overlay:** Real-time feedback on FPS, GPU/CPU usage, and active modes.

---

## Installation & Setup

### 1. Prerequisites
Make sure you have Python 3.8+ installed. 

### 2. Download the Models
Due to GitHub file size limits, the deep learning models are not included in this repository. You must download them manually before running the script:

1. View the download links and instructions in [1/models.txt](1/models.txt).
2. Download both files:
   - `inswapper_128.onnx`
   - `gfpgan_1.4.onnx`
3. Place them directly in the `1/` directory alongside `face_detector.py`.

### 3. Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```
*(Note: If you have an NVIDIA GPU, make sure you install `onnxruntime-gpu` to get high FPS).*

### 4. Run the Application
Run the face detector script:
```bash
python 1/face_detector.py
```

---

## Controls

When the camera window is active, use the following keys to control the application:

| Key | Action |
|---|---|
| **`L`** | Load a source face photo (opens a file browser dialog) |
| **`R`** | Reset/Clear the loaded photo |
| **`E`** | Toggle HQ (GFPGAN) Enhancement Mode |
| **`Q` / `ESC`** | Quit the application |
