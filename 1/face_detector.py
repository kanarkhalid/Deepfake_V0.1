"""
                     kanar khalid
                    f ace swap app
Face Swap App — Load a Photo and Swap It On Your Live Face
============================================================
Uses InsightFace (deep learning) for realistic face swapping.
Load a photo of someone and their face appears on yours in real-time.

Controls:
    L      - Load a face photo
    R      - Reset / clear loaded photo
    Q/ESC  - Quit
"""

import os
import sys
import site

_user_site = site.getusersitepackages()
_nvidia_base = os.path.join(_user_site, "nvidia")
if os.path.isdir(_nvidia_base):
    for _sub in os.listdir(_nvidia_base):
        _bin = os.path.join(_nvidia_base, _sub, "bin")
        if os.path.isdir(_bin):
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
            # Also register as DLL directory (Python 3.8+)
            try:
                os.add_dll_directory(_bin)
            except OSError:
                pass

#  ML libraries ─────────────────────────────────────────────────
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import onnxruntime as ort
import tkinter as tk
from tkinter import filedialog
import urllib.request
import time

# ── Verify GPU ───────────────────────────────────────────────────────────────
available = ort.get_available_providers()
print(f"ONNX Runtime providers: {available}")
if 'CUDAExecutionProvider' in available:
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("✓ GPU (CUDA) will be used — expect high FPS!")
else:
    PROVIDERS = ['CPUExecutionProvider']
    print("✗ WARNING: CUDA not available, falling back to CPU (will be slow)")

# ── Setup InsightFace ────────────────────────────────────────────────────────
print("Loading face detection models...")
app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)

app.prepare(ctx_id=0, det_size=(320, 320))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inswapper_128.onnx")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    print("Please download inswapper_128.onnx and place it in the same folder.")
    sys.exit(1)

print("Loading face swap model...")
swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=PROVIDERS)

if hasattr(swapper, 'session') and hasattr(swapper.session, 'get_providers'):
    print(f"Swapper active providers: {swapper.session.get_providers()}")

# Enhancement ───────────────────────────────
GFPGAN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfpgan_1.4.onnx")
GFPGAN_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"


def download_file(url, dest):
    print(f"Downloading GFPGAN model to {dest}...")
    start_time = time.time()
    
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            mb_downloaded = read_so_far / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = time.time() - start_time
            speed = mb_downloaded / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) | Speed: {speed:.2f} MB/s")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded: {read_so_far / (1024 * 1024):.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print("\nDownload completed successfully!")


if not os.path.exists(GFPGAN_MODEL_PATH):
    try:
        download_file(GFPGAN_URL, GFPGAN_MODEL_PATH)
    except Exception as e:
        print(f"\nERROR downloading GFPGAN: {e}")
        print("Please download it manually from: https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx")
        sys.exit(1)

print("Loading GFPGAN model...")
gfpgan_session = ort.InferenceSession(GFPGAN_MODEL_PATH, providers=PROVIDERS)
gfpgan_input_name = gfpgan_session.get_inputs()[0].name
print("GFPGAN loaded!")

print("Models loaded!")


def load_source_image():
    """Open file dialog to select a face photo."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select a face photo to swap",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
    )
    root.destroy()
    return path


def enhance_and_paste_back(img, swapper, cam_face, source_face, gfpgan_session, gfpgan_input_name):
    """Perform high-quality 512x512 face swapping using GFPGAN enhancement."""
    # 1. Get the face in 128x128 
    bgr_fake, M = swapper.get(img, cam_face, source_face, paste_back=False)
    
    # 2. Resize the fake face to 512x512 for GFPGAN input
    face_512 = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_CUBIC)
    
    # 3. Preprocess for GFPGAN: BGR to RGB, normalise to [-1, 1], CxHxW
    img_rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    input_data = img_norm.transpose(2, 0, 1)
    input_data = np.expand_dims(input_data, axis=0)
    
    # 4. Run GFPGAN inference
    output = gfpgan_session.run(None, {gfpgan_input_name: input_data})
    
    # 5. Postprocess:denormalise RGB to BGR
    result = output[0][0].transpose(1, 2, 0)
    result = (result * 0.5 + 0.5) * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    bgr_fake_512 = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # 6. Aligned crop of original face at 512x512 resolution for background subtraction
    aimg_512, _ = face_align.norm_crop2(img, cam_face.kps, 512)
    
    # 7. Scale transformation matrix M by 4x for the 512x512 canvas
    M_512 = M * 4.0
    
    # 8. Blending and pasting back
    target_img = img.copy()
    fake_diff = bgr_fake_512.astype(np.float32) - aimg_512.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    
    # Border masking to prevent harsh edges
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    
    IM = cv2.invertAffineTransform(M_512)
    img_white = np.full((512, 512), 255, dtype=np.float32)
    
    bgr_fake_warped = cv2.warpAffine(bgr_fake_512, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white_warped = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff_warped = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    
    img_white_warped[img_white_warped > 20] = 255
    fthresh = 10
    fake_diff_warped[fake_diff_warped < fthresh] = 0
    fake_diff_warped[fake_diff_warped >= fthresh] = 255
    
    img_mask = img_white_warped
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        kernel = np.ones((2, 2), np.uint8)
        fake_diff_warped = cv2.dilate(fake_diff_warped, kernel, iterations=1)
        
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff_warped = cv2.GaussianBlur(fake_diff_warped, blur_size, 0)
        
        img_mask /= 255.0
        fake_diff_warped /= 255.0
        
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * bgr_fake_warped + (1.0 - img_mask) * target_img.astype(np.float32)
        return fake_merged.astype(np.uint8)
        
    return img


def draw_hud(frame, w, h, loaded, fps, gpu_active, hq_mode):
    """Draw minimal HUD overlay with HQ and GPU indicators."""
    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    if loaded:
        txt = "FACE SWAP ACTIVE"
        if hq_mode:
            txt += " — HQ Enhanced"
        else:
            txt += " — Standard Mode"
        col = (0, 255, 120) if hq_mode else (0, 255, 230)
    else:
        txt = "Press L to load a face photo"
        col = (0, 170, 255)
    cv2.putText(frame, txt, (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, col, 2, cv2.LINE_AA)

    # FPS + GPU + HQ indicators
    gpu_txt = "GPU" if gpu_active else "CPU"
    gpu_col = (0, 255, 0) if gpu_active else (0, 0, 255)
    
    hq_txt = "HQ:ON" if hq_mode else "HQ:OFF"
    hq_col = (0, 255, 0) if hq_mode else (128, 128, 128)
    
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 240, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{gpu_txt}]", (w - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"[{hq_txt}]", (w - 75, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hq_col, 2, cv2.LINE_AA)

    # Bottom bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "L:Load Photo   R:Reset   E:Toggle HQ   Q:Quit",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (190, 190, 190), 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    gpu_active = 'CUDAExecutionProvider' in PROVIDERS
    hq_mode = gpu_active  

    print("=" * 50)
    print(f"  FACE SWAP APP ({'GPU' if gpu_active else 'CPU'} Mode)")
    print("  Press L to load a face photo")
    print("  Press E to toggle HQ (GFPGAN) Mode")
    print("  Press Q to quit")
    print("=" * 50)

    source_face = None
    source_img = None
    fps = 0
    prev_time = cv2.getTickCount()

    # Performance: skip detection on most frames, reuse cached faces
    cached_cam_faces = []
    frame_count = 0
    DETECT_EVERY_N = 3 

# if no face exist :
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        img = cv2.imread(sys.argv[1])
        if img is not None:
            faces = app.get(img)
            if faces:
                source_face = faces[0]
                source_img = img
                print(f"Source face loaded from: {sys.argv[1]}")
            else:
                print("No face found in that image!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
        prev_time = current_time

        # Detect faces only every N frames to save processing time
        frame_count += 1
        if frame_count % DETECT_EVERY_N == 0:
            cached_cam_faces = app.get(frame)

        # Perform face swap using cached face positions
        if source_face is not None and cached_cam_faces:
            for cam_face in cached_cam_faces:
                try:
                    if hq_mode:
                        frame = enhance_and_paste_back(frame, swapper, cam_face, source_face, gfpgan_session, gfpgan_input_name)
                    else:
                        frame = swapper.get(frame, cam_face, source_face, paste_back=True)
                except Exception as e:
                    cv2.putText(frame, f"Swap error: {str(e)[:50]}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Source thumbnail
        if source_img is not None:
            th = 100
            tw = int(source_img.shape[1] * th / source_img.shape[0])
            thumb = cv2.resize(source_img, (tw, th))
            cv2.rectangle(thumb, (0, 0), (tw - 1, th - 1), (0, 255, 100), 2)
            xo = w - tw - 10
            yo = 48
            if xo > 0 and yo + th < h:
                frame[yo:yo + th, xo:xo + tw] = thumb

        draw_hud(frame, w, h, source_face is not None, fps, gpu_active, hq_mode)
        cv2.imshow("Face Swap", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("l"), ord("L")):
            path = load_source_image()
            if path:
                img = cv2.imread(path)
                if img is not None:
                    faces = app.get(img)
                    if faces:
                        source_face = faces[0]
                        source_img = img
                        print(f"Source face loaded from: {path}")
                    else:
                        print("No face found in that image! Try another.")
                else:
                    print(f"Could not read image: {path}")
        elif key in (ord("r"), ord("R")):
            source_face = None
            source_img = None
            print("Source photo cleared.")
        elif key in (ord("e"), ord("E")):
            hq_mode = not hq_mode
            print(f"HQ mode changed to: {hq_mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# By kanar khalid