<div align="center">

# 🎭 Deepfake Face Swap

### Real-Time AI Face Swapping with GPU Acceleration

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-GPU-FF6F00?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![InsightFace](https://img.shields.io/badge/InsightFace-Deep_Learning-E34F26?style=for-the-badge)](https://insightface.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*Load a photo of anyone and watch their face appear on yours — in real time.*

</div>

---

## ✨ Features

| Feature | Description |
|:---:|---|
| 🎥 **Real-Time Swap** | High-performance face detection and replacement directly on your webcam stream |
| 🔬 **HQ Enhancement** | GFPGAN-powered 512×512 upscaling restores pores, eyes, and fine facial details |
| ⚡ **GPU Accelerated** | Automatic NVIDIA CUDA detection via ONNX Runtime for smooth, high-FPS output |
| 🖥️ **Live HUD** | On-screen overlay showing FPS, GPU/CPU status, and active mode indicators |
| 📸 **Photo Loading** | Simple file browser to load any face photo (JPG, PNG, BMP, WebP) |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kanarkhalid/Deepfake_V0.1.git
cd Deepfake_V0.1
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python numpy insightface onnxruntime-gpu
```

> 💡 **No NVIDIA GPU?** Use `onnxruntime` instead of `onnxruntime-gpu`. It will run on CPU (slower).

### 3️⃣ Download the Models

The AI models are too large for GitHub (~900 MB total). Download them manually:

<table>
  <tr>
    <th>Model</th>
    <th>Size</th>
    <th>Download</th>
  </tr>
  <tr>
    <td><b>🧠 inswapper_128.onnx</b><br><sub>InsightFace Face Swap</sub></td>
    <td>~554 MB</td>
    <td>
      <a href="https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx">📥 GitHub Release</a><br>
      <a href="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx">📥 Hugging Face</a>
    </td>
  </tr>
  <tr>
    <td><b>✨ gfpgan_1.4.onnx</b><br><sub>GFPGAN Face Enhancement</sub></td>
    <td>~340 MB</td>
    <td>
      <a href="https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx">📥 Hugging Face</a>
    </td>
  </tr>
</table>

> ⚠️ **Place both `.onnx` files in the project root directory** (same folder as `face_detector.py`).

### 4️⃣ Run

```bash
python face_detector.py
```

---

## 🎮 Controls

<div align="center">

| Key | Action |
|:---:|---|
| `L` | 📂 Load a source face photo |
| `R` | 🔄 Reset / clear loaded photo |
| `E` | ✨ Toggle HQ Enhancement (GFPGAN) |
| `Q` / `ESC` | 🚪 Quit the application |

</div>

---

## 🏗️ How It Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐
│  Webcam Feed │───▶│ InsightFace  │───▶│  inswapper_128  │───▶│  Output  │
│   (OpenCV)   │    │  Detection   │    │   Face Swap     │    │  Frame   │
└─────────────┘    └──────────────┘    └────────┬────────┘    └──────────┘
                                                │
                                       ┌────────▼────────┐
                                       │  GFPGAN (HQ)    │
                                       │  Enhancement    │
                                       │  512×512        │
                                       └─────────────────┘
```

1. **Face Detection** — InsightFace (`buffalo_l`) detects and locates faces in each webcam frame
2. **Face Swap** — `inswapper_128` replaces the detected face with the source photo's face
3. **HQ Enhancement** *(optional)* — GFPGAN upscales the swapped face to 512×512, restoring fine details
4. **Blending** — The enhanced face is seamlessly blended back into the original frame

---

## 📋 Requirements

- **Python** 3.8+
- **Webcam** (built-in or USB)
- **NVIDIA GPU** *(recommended)* with CUDA for real-time performance
- **~900 MB** disk space for models

---

## 📁 Project Structure

```
Deepfake_V0.1/
├── face_detector.py     # Main application
├── models.txt           # Model download links & instructions
├── README.md            # This file
├── .gitignore           # Excludes models & caches from Git
├── inswapper_128.onnx   # ⬇️ Download manually (not in repo)
└── gfpgan_1.4.onnx      # ⬇️ Download manually (not in repo)
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. The creator is not responsible for any misuse of this technology. Always obtain consent before using someone's likeness and adhere to all applicable laws and ethical guidelines regarding AI-generated content.

---

<div align="center">

**Made with ❤️ by [kanarkhalid](https://github.com/kanarkhalid)**

</div>
