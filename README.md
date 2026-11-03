# Thermal Object Detection from a Drone

A real-time system that detects and classifies objects (people, cars, street lamps) at night using a **thermal camera mounted on a drone**, an **analog video link** to the ground, and a **YOLO deep-learning model** shown in a **desktop app**.

---

## The problem

Most object-detection models work on regular color (RGB) cameras, and those cameras fail in the dark. Thermal cameras see heat instead of light, so they keep working at night. Thermal images are different from color images, though: they have no color, a lower resolution, and contrast that changes with the temperature. Off-the-shelf YOLO models trained on color photos don't handle them well.

Security tasks such as watching borders, protecting forces in the field, and scanning roads often need a drone operator watching a screen for hours. A system that detects objects on its own and alerts the operator reduces that load.

## Our solution

```
 ┌──────────── Airborne unit (168 g) ────────────┐        ┌──────────── Ground station ────────────┐
 │ Thermal camera ──► Analog video transmitter ──┼─ RF ──►│ USB receiver ──► Laptop ──► YOLO App   │
 │ (LWIR, 256x192, 25 FPS)   + own LiPo battery  │        │ (seen as a webcam)   (detect + display)│
 └───────── 3D-printed mount on DJI Phantom 4 ───┘        └────────────────────────────────────────┘
```

- **Hardware:** an uncooled LWIR thermal camera, an analog transmitter and a separate battery sit in a 3D-printed mount on the drone's landing gear. They don't touch the drone's own wiring. Analog transmission was chosen for its very low delay.
- **Ground station:** a USB receiver shows up on the laptop as an ordinary webcam.
- **Software:** a Python app runs a YOLO model on every frame. It draws bounding boxes with confidence scores, and the operator can change the detection threshold live.
- **Model:** **YOLO26s**, trained on our own night-time thermal dataset (213 images, 3 classes: `person`, `car`, `streetlamp`). The images were taken from 30 to 45 m altitude at a 45° camera angle.

### Final results (our model, night test set)

| Metric | Value |
|---|---|
| Precision | **0.933** |
| Recall | **0.911** |
| mAP50 | **0.919** |
| mAP50-95 | 0.486 |
| Processing time per frame | ~36 ms (fits the 40 ms budget for 25 FPS) |

> **Limitation:** the model was trained on night images only. In daytime the ground heats up and the thermal contrast flips ("thermal contrast reversal"), so performance drops sharply (mAP50 ≈ 0.34).

---

## Repository structure

```
Experiments/
├── EXP_1_RGB_pretrained_datasets/   Experiment 1: compare YOLO versions in MATLAB on color images
├── EXP_2_UAV_trained_datasets/      Experiment 2: train on a public thermal drone dataset (HIT-UAV)
│   └── python/                      the YOLO26 part of experiment 2
├── EXP_3_Custom_trained_datasets/   Experiment 3: build our own thermal dataset and train the final model
└── YOLO app/                        The desktop application
```

---

## The experiments

The project went through three experiments. Each one led to the next.

### Experiment 1: Comparing YOLO versions on RGB images (MATLAB)

**Goal:** find which YOLO version gives the best balance of accuracy and speed, using models pre-trained on the COCO dataset.
**Tested:** YOLOv2, YOLOv3, YOLOv4 and YOLOX on 265 color images.
**Result:** YOLOX was the most accurate (mAP50 = 0.859). Every MATLAB model was too slow for real time (2 to 14 FPS).

| File | What it does |
|---|---|
| `exp1_YOLOv2_RGB_pretrained_test.m` | Loads YOLOv2, runs it on the test images and computes precision, recall, F1 and the confusion matrix |
| `exp1_YOLOv3_RGB_pretrained_test.m` | The same, for YOLOv3 |
| `exp1_YOLOv4_RGB_pretrained_test.m` | The same, for YOLOv4 |
| `exp1_YOLOx_RGB_pretrained_test.m` | The same, for YOLOX |
| `exp1_model_compare_RGB_pretrained_test.m` | Loads the saved results of all four models and plots them side by side (precision-recall curves, F1, run time) |
| `yoloformat2trainingData.m` | Helper: reads a folder of images and YOLO `.txt` labels and builds a MATLAB table |
| `ConvertYoloFormat2MatlabFormat.m` | Helper: converts one box from YOLO format (normalized center, width, height) to MATLAB format (pixel corner, width, height) |

### Experiment 2: Training on a public thermal dataset (HIT-UAV)

**Goal:** train models on real thermal drone footage and compare them.
**Dataset:** HIT-UAV, about 5,800 thermal images with the classes person, car, bicycle/motorcycle, other vehicle and "don't care".
**Tested:** YOLOv4 and YOLOX trained in MATLAB, and YOLO26n trained in Python on Google Colab.
**Result:** YOLO26n won clearly (mAP50 = 0.781 at 23.8 FPS, against 0.506 for YOLOv4 at 2.3 FPS). This convinced us to **move from MATLAB to Python**.

| File | What it does |
|---|---|
| `exp2_YOLOv4_UAV_trained_dataset.m` | Trains YOLOv4 on HIT-UAV in MATLAB and evaluates it |
| `exp2_YOLOx_UAV_trained_dataset.m` | Trains YOLOX on HIT-UAV in MATLAB and evaluates it |
| `exp2_yoloformat2trainingData_UAV_trained_dataset.m` | Helper: converts the YOLO-format dataset into MATLAB tables (train / val / test) |
| `python/exp2_YOLO26_UAV_trained_dataset.ipynb` | Google Colab notebook: trains a YOLO model on HIT-UAV with the Ultralytics library |
| `python/exp2_YOLO26_check_UAV_trained_dataset.py` | Plays a video, runs the trained model on each frame and shows the boxes and FPS on screen (press **Esc** to quit) |
| `python/best_yolo26_2.pt` | The trained weights from this experiment |

### Experiment 3: Our own thermal dataset and the final model

**Goal:** public datasets didn't match our camera, altitude and setting, so we built our own.
**Steps:**
1. Recorded night flights with our system and extracted frames from the videos.
2. Picked useful frames by hand and dropped blurry or duplicate ones. That left 213 images.
3. Labeled them with the open-source **image-annotator** tool and exported the labels as COCO JSON.
4. Converted the labels to YOLO format and split the images 70% train, 15% validation, 15% test.
5. Trained two models on Google Colab (GPU): **YOLO26n** (model 1) and **YOLO26s** (model 2, the final one).

| File | What it does |
|---|---|
| `COCO2YOLO.py` | Converts the COCO JSON labels from the annotator into YOLO `.txt` labels, randomly splits the images into train / val / test folders and writes the `data.yaml` file that YOLO training needs |
| `FormatChanging_check.py` | Sanity check: draws the converted YOLO boxes back onto the images so you can confirm the conversion is correct |
| `exp_3_train_notebook_Custom_trained_datasets.ipynb` | Google Colab notebook: trains YOLO26s with tuned settings, then compares model 1 and model 2 on the test set (confusion matrix, precision, recall, mAP) |
| `How to Download image annotator` | Step-by-step notes for installing the labeling tool (Anaconda plus `digitalsreeni-image-annotator`) |
| `yolo26n_best_weights.pt` | Trained weights, model 1 (YOLO26n, faster) |
| `yolo26s_best_weights.pt` | Trained weights, model 2 (YOLO26s, more accurate). **This is the final model.** |

---

## The YOLO App

A desktop application that plays a live camera feed or a video file and runs a YOLO model on it in real time.

### Code overview

| File | Role |
|---|---|
| `app.py` | Entry point. Opens the main window and switches between the two screens |
| `setup_page.py` | The first screen: choose the model and the input |
| `player_page.py` | The main screen: the video, the top and bottom control bars, recording and the detection log |
| `engine.py` | Background work: loads the model, reads frames from the camera or file, runs detection, records video and writes the log |
| `draw.py` | Draws the boxes, labels, confidence, FPS and object count on each frame |
| `style.py` | The app's visual theme |
| `requirements.txt` | Python libraries the app needs |
| `run.command` | Launcher script for macOS |
| `yolo26s_best_weights.pt` | **Our trained thermal model** (person / car / streetlamp) |
| `yolo26n.pt` | The default general-purpose YOLO model (80 everyday classes, for normal color video) |

### Installation and running

**Requirements:** Python 3.10 or newer, and an internet connection for the first installation.

The app is a Python package, so it has to be started as a module named `yolo_app`. **First rename the folder `YOLO app` to `yolo_app`.** Then open a terminal in the folder that *contains* it (`Experiments`).

**Windows (Command Prompt):**
```bat
cd D:\Users\Oren\Downloads\Experiments
python -m venv .venv
.venv\Scripts\pip install -r yolo_app\requirements.txt
.venv\Scripts\python -m yolo_app
```

**macOS (Terminal):**
```bash
cd path/to/Experiments
python3 -m venv .venv
.venv/bin/pip install -r yolo_app/requirements.txt
.venv/bin/python -m yolo_app
```

The installation step runs only once and takes a few minutes. After that, only the last command is needed to open the app.

### How to use it

**1. Setup screen**

- **Model:** click **Browse…** and pick a model file, then click **Load**.
  - For thermal video, use **`yolo26s_best_weights.pt`** (our model).
  - For regular color video, `yolo26n.pt` works.
  - When the model loads, a green line shows its name, the number of classes and the hardware it runs on (GPU or CPU).
- **Input:**
  - **Video file:** pick a recorded video (`.mp4`, `.avi`, `.mov`, …).
  - **Camera:** live feed. Choose the camera number: `0` is usually the laptop's built-in camera, and the drone receiver is usually `1`.
- Click **Start**.

The app remembers your last model and input for next time.

**2. Main screen**

The video plays with a box around each detected object, showing the class name and confidence. The corner shows the FPS and the number of objects in view.

*Top bar:*

| Control | What it does |
|---|---|
| **← Settings** | Go back to the setup screen |
| **Objects ▾** | Choose which classes to show (for example, only `person`) |
| **View ▾** | Turn on or off: boxes, class name, confidence, FPS, object count |
| **Confidence ▾** | The detection threshold (0 to 1, default 0.35). Lower values find more objects but give more false alarms. Higher values give fewer, surer detections |
| **Detect / Track** | *Detect* finds objects in each frame. *Track* also gives each object an ID and follows it between frames |
| **● Rec** | Record the video with the boxes drawn on it |
| **Log (N)** | Open the detection log: a table of each object that appeared, its class, the time and its confidence. It can be saved as CSV |

*Bottom bar:*

- **Video file:** play or pause, jump 5 s back or forward, and drag the time slider.
- **Live camera:** you can rewind up to the last 30 seconds. Click **● LIVE** to return to the live picture.
- **Keyboard:** `Space` plays or pauses, and `←` / `→` jump 5 seconds.

**Where files are saved:**

| | Location |
|---|---|
| Recordings | `<your user folder>/Movies/YOLO App/` |
| Detection logs (saved automatically) | `<your user folder>/Documents/YOLO App/logs/` |

---

## Tools and technologies

- **Languages:** Python, MATLAB
- **Deep learning:** Ultralytics YOLO (YOLO26n / YOLO26s), MATLAB YOLOv2/v3/v4/X
- **App:** PySide6 (Qt), OpenCV, ByteTrack (tracking)
- **Training:** Google Colab with an NVIDIA T4 GPU
- **Labeling:** image-annotator (COCO JSON exported, then converted to YOLO)
- **Hardware:** DJI Phantom 4, uncooled VOx thermal camera (LWIR, 256x192, 25 FPS), analog video transmitter and USB (UVC) receiver, 3D-printed mount

## Future improvements

- A larger and more varied dataset, with day images, different seasons and more classes (animals, motorcycles, trucks)
- Separate day and night models, chosen automatically
- Adding an RGB camera next to the thermal camera
- A higher-resolution thermal sensor and a motorized gimbal
- Running the model on a small computer onboard the drone
