# DanceDanceMid‑Solution
A multi‑camera, deep‑learning re‑imagining of Dance Dance Revolution that turns any floor into a playable stage.

> Pipeline Four RTSP cameras → synchronized frame pipeline → 3‑frame GIFs → R(2 + 1)D + FC neural network → JSON socket stream → PyGame DDR client.  
> Everything is wired together with Python, FFmpeg, and a Selenium helper that auto‑builds step charts from any song.

---

## AUTHORS
Eliot Weiner · Humzah Durrani · Sadman Kabir · Benjamin Hsu · Arthur Wang · Anton Beca

---

## TABLE OF CONTENTS
1. Project Highlights  
2. System Overview  
3. Hardware Setup  
4. Software Stack  
5. Installation  
6. Quick Start  
7. Dataset & Training  
8. Repository Layout  
9. Troubleshooting  
10. Contributing  
11. License  

---

## 1  PROJECT HIGHLIGHTS
* Camera‑only gameplay—tape a 4‑square board on the floor; vision does the rest.  
* Barrier‑synchronized FFmpeg threads keep all cameras in lock‑step.  
* One‑command song importer wraps **Dance Dance Convolution** to create `.sm` charts from any MP3 or YouTube link.  
* R(2 + 1)D backbone in half‑precision sustains ~20 FPS on a single consumer GPU.  
* PyGame GUI replicates classic DDR visuals and consumes model output via a non‑blocking socket.  
* Modular codebase: separate modules for data capture, annotation, training, and the production `MAIN_GAME_LOOP`.

(Add your system diagram and gameplay GIF in `docs/images/` and reference them here.)

---

## 2  SYSTEM OVERVIEW
```
┌───────────────┐      RTSP (TCP)      ┌────────────────┐
│ 4× IP Cameras │ ─────────►────────── │  Frame Threads │
└───────────────┘                      └───────┬────────┘
                                               ▼
                                  ┌─────────────────────┐
                                  │ Rolling 3‑Frame GIF │
                                  │  + Cropping + Norm  │
                                  └─────────┬───────────┘
                                            ▼
                        ┌────────────────────────────────────┐
                        │  R(2+1)D‑18 Backbone (+ concat)    │
                        │       → Fully Connected Heads      │
                        └─────────┬──────────────────────────┘
                                  ▼
                           JSON over TCP Socket
                                  ▼
                         ┌─────────────────┐
                         │ PyGame DDR GUI  │
                         └─────────────────┘
```
Full technical background—including data governance and results—is in `docs/EC535 Final Project Technical Report.pdf`.

---

## 3  HARDWARE SETUP

| Item                                | Purpose                     |
|-------------------------------------|-----------------------------|
| 4 × PoE IP Cameras (Axis P5512)     | Multi‑angle foot tracking   |
| Gigabit PoE Switch                  | Dedicated LAN, low jitter   |
| NVIDIA GPU (≥ RTX 4070 Ti Super)    | Real‑time inference         |
| Floor markings (tape/chalk)         | Simple 4‑square dance pad   |

---

## 4  SOFTWARE STACK

| Layer      | Tech                                           |
|------------|-----------------------------------------------|
| Core ML    | PyTorch ≥ 2.2, torchvision, torchmetrics      |
| Vision I/O | numpy, Pillow, opencv‑python                  |
| Media      | FFmpeg (CLI), ffmpeg‑python (optional)        |
| GUI        | pygame                                        |
| OS deps    | Python 3.10+, Git, Google Chrome/Chromium     |

`requirements.txt` lists exact versions.

---

## 5  INSTALLATION

### 5.1  Clone & Virtual Env
```
git clone https://github.com/ElliotWeiner/DanceDanceMid-Solution.git
cd DanceDanceMid-Solution

python -m venv .venv
# Windows: .venv\Scriptsctivate
source .venv/bin/activate
```

### 5.2  Python Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3  System Dependencies
* **FFmpeg** – download from <https://ffmpeg.org> and ensure `ffmpeg` is on your `PATH`.  
* **Google Chrome** – latest stable release.  
* **ChromeDriver** – auto‑installed by `webdriver‑manager` or download manually.  
* **CUDA** – if using GPU, install matching NVIDIA drivers and CUDA 11.8+.

### 5.4  Camera Config
Edit the `cameras` dictionary in `MAIN_GAME_LOOP/cameras.py` (or the capture script):
```python
cameras = {
    2: "rtsp://user:pass@192.168.1.131/axis-media/media.amp",
    3: "rtsp://user:pass@192.168.1.160/axis-media/media.amp",
}
```

### 5.5  Pre‑Trained Weights (optional)
Place `feet_net_final.pth` in `model/training/` to skip training.

---

## 6  QUICK START

1. **Start the Model Socket Server**
   ```
   python MAIN_GAME_LOOP/model_socket_server.py
   ```
2. **Launch the DDR GUI**
   ```
   python MAIN_GAME_LOOP/ddr_gui.py
   ```
3. **(Optional) Import a Custom Song**
   ```
   python scripts/song_importer.py        --youtube "https://youtu.be/dQw4w9WgXcQ"        --title "Never Gonna Give You Up"        --artist "Rick Astley" --difficulty 5
   ```
4. **Dance!**  
   Select the song in‑game and step on your taped arrows—the model presses the keys in real time.

---

## 7  DATASET & TRAINING

* **Recording** – `scripts/record_multicam.py` captures synchronized MP4s.  
* **Annotation** – `scripts/chunk_and_label.py` converts footage to 3‑frame GIFs and CSV labels.  
* **Training** – `model/train.py` (SGD + momentum, cyclical LR, class weighting).  
* **Metrics** – Accuracy logged to TensorBoard; samples live under `runs/`.

(The 76 GB dataset is not included; contact the authors for access.)

---

## 8  REPOSITORY LAYOUT
```
.
├── MAIN_GAME_LOOP/          ← production inference & GUI
├── model/                   ← training code & checkpoints
├── scripts/                 ← utilities (recording, annotation, song import)
├── graphics/                ← PyGame assets + StepMania songs
├── docs/                    ← reports, diagrams, README images
└── dataset/                 ← (git‑ignored) large training data
```

---

## 9  TROUBLESHOOTING

| Symptom                                          | Fix                                             |
|--------------------------------------------------|-------------------------------------------------|
| `FFmpeg error: Unrecognized option 'stimeout'`   | Update to FFmpeg ≥ 6.1                          |
| GUI freezes on start                             | Start the model socket server before the GUI    |
| Cameras drift out of sync                        | Keep them on an isolated PoE switch; barrier    |
|                                                  | resets every 100 frames in code                 |
| `ModuleNotFoundError: ddr_gui`                   | Run scripts from repo root or set PYTHONPATH    |

---

## 10  CONTRIBUTING
Pull requests are welcome—please open an issue first for major changes.

---

## 11  LICENSE
MIT © 2025 Eliot Weiner, Humzah Durrani, Sadman Kabir, Benjamin Hsu, Arthur Wang, Anton Beca
