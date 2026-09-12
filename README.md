# Smart Surveillance for Intrusion Detection using Background Subtraction

A BTech Computer Vision mini-project that detects moving objects in a fixed-camera
scene (webcam or CCTV-style video) and raises an **"INTRUSION DETECTED"** alert when
a moving object enters a user-defined restricted zone.

Core technique: **OpenCV background subtraction (MOG2 / KNN)** — no deep learning,
no YOLO, no face recognition.

## Project Structure

```
opencv-intrusion-detection/
├── intrusion_detection.py   # main runnable script (the whole project)
├── requirements.txt         # Python dependencies
├── intruder.mp4             # sample CCTV-style test video
├── report/
│   ├── REPORT.md            # academic report content
│   ├── VIVA_QUESTIONS.md    # viva / oral exam preparation
│   └── PRESENTATION_SCRIPT.md  # 1-minute demo video script
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

(or directly: `pip install opencv-python numpy`)

Requires Python 3.8+.

## How to Run

Run on the bundled sample video (default):

```bash
python intrusion_detection.py
```

Run using your webcam instead:

```bash
python intrusion_detection.py --source 0
```

Run on your own video file:

```bash
python intrusion_detection.py --source path/to/your_video.mp4
```

Use KNN instead of MOG2 for background subtraction:

```bash
python intrusion_detection.py --algo KNN
```

Two windows will open:
- **Live Feed** — the video with bounding boxes, the restricted zone, and the
  intrusion alert banner.
- **Foreground Mask** — the raw output of background subtraction, useful for
  explaining how the algorithm sees "motion".

Press **q** or **ESC** in either window to exit.

## How It Works (short version)

1. Each frame is compared against a running statistical model of the background
   (`cv2.createBackgroundSubtractorMOG2`) to produce a foreground mask.
2. The mask is thresholded to drop shadow pixels, then cleaned up with
   morphological opening + dilation to remove noise and solidify object shapes.
3. `cv2.findContours` finds the outlines of remaining foreground blobs; small
   ones are discarded as noise.
4. A bounding box is drawn around each real moving object.
5. If a bounding box overlaps the rectangular restricted zone drawn on screen,
   the frame is flagged as an intrusion and "INTRUSION DETECTED" is displayed.

Full explanation, architecture diagram, and report material are in `report/REPORT.md`.

## Adjusting the Restricted Zone / Sensitivity

Open `intrusion_detection.py` and edit the constants near the top of the file:

- `ZONE_X1, ZONE_Y1, ZONE_X2, ZONE_Y2` — restricted area corners (as a fraction
  of the frame width/height, so `0.5, 0.5` is always the center of the frame).
- `MIN_CONTOUR_AREA` — increase this if small movements (leaves, noise, shadows)
  are wrongly triggering detections; decrease it to detect smaller/farther objects.

## Troubleshooting

See the **Troubleshooting** section in the full write-up (`report/REPORT.md`) for
fixes to common issues: camera not opening, noisy detections, constant bounding
boxes, and OpenCV installation errors.
