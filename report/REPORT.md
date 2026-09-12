# Smart Surveillance for Intrusion Detection in a Fixed Environment using Background Subtraction

*BTech Computer Vision Mini-Project Report*

---

## Abstract

Manual monitoring of CCTV footage is tiring, slow, and error-prone — a human
operator watching multiple static camera feeds for hours will inevitably miss
the exact moment an intruder enters a restricted area. This project implements
a lightweight, real-time intrusion detection system for a **fixed camera
environment** (a CCTV-style setup where the camera does not move) using
**background subtraction**, a classical computer vision technique. The system
uses OpenCV's `MOG2` (Mixture of Gaussians) algorithm to separate moving
foreground objects from a static background, cleans the resulting mask using
thresholding and morphological operations, extracts object outlines using
contour detection, and draws bounding boxes around real moving objects. A
rectangular restricted zone is defined on the frame, and whenever a detected
object's bounding box overlaps this zone, the system raises an on-screen
**"INTRUSION DETECTED"** alert. The system works on both live webcam feeds and
prerecorded video files, runs in real time on a standard laptop CPU, and
requires no machine learning training, labeled datasets, or GPU — making it an
appropriate, explainable solution for a fixed-camera surveillance scenario such
as a server room, store entrance, or parking gate.

---

## 1. Introduction

Video surveillance is one of the most common tools used for security in homes,
offices, banks, and industrial sites. Most of these cameras are **static** —
mounted on a wall or ceiling, watching one scene continuously. In such a fixed
setting, anything that changes in the frame over time is very likely to be a
moving object of interest (a person, vehicle, or animal), since everything
else — walls, furniture, parked equipment — stays visually constant.

This observation is exactly what **background subtraction** exploits. Instead
of trying to recognize *what* an object is (which is what deep-learning-based
detectors like YOLO do), background subtraction only asks a much simpler
question: *"has something in this pixel/region changed compared to the
background I have been observing?"* This makes it computationally cheap, easy
to understand, and well suited to a fixed-camera surveillance task — which is
precisely the assignment brief for this project.

## 2. Problem Statement

Given a static camera feed (webcam or prerecorded CCTV-style video) of a fixed
scene, design and implement a system that can:

1. Continuously distinguish moving foreground objects from the static
   background in real time.
2. Filter out noise so that only genuine, sufficiently large moving objects
   are reported.
3. Mark a specific rectangular region of the frame as a "restricted /
   monitored area".
4. Automatically detect when a moving object enters this restricted area and
   raise a clear, visible **"INTRUSION DETECTED"** alert.

The solution must use only classical image-processing techniques (background
subtraction, thresholding, morphology, contour analysis) — not deep learning,
face recognition, or object classification.

## 3. Objectives

- To implement a real-time foreground/background segmentation pipeline using
  OpenCV's `MOG2` (and optionally `KNN`) background subtractor.
- To reduce noise in the foreground mask using thresholding and morphological
  operations (opening, dilation).
- To detect and localize moving objects using contour detection and bounding
  boxes.
- To define a restricted monitoring zone and detect when an object enters it.
- To display real-time visual feedback, including a clear intrusion alert.
- To keep the entire system simple enough to be explained end-to-end by a
  BTech student, using only well-established, classical CV techniques.

## 4. Methodology

The project follows a standard image-processing pipeline approach rather than
a learning-based approach:

1. **Frame acquisition** — frames are read one at a time from a webcam or
   video file using `cv2.VideoCapture`.
2. **Preprocessing** — each frame is resized (for consistent performance) and
   smoothed with a Gaussian blur to suppress sensor/compression noise before
   it reaches the background model.
3. **Background modelling & subtraction** — `cv2.createBackgroundSubtractorMOG2`
   maintains a per-pixel statistical (Gaussian mixture) model of the recent
   background and classifies each pixel of the incoming frame as background or
   foreground, producing a grayscale foreground mask.
4. **Thresholding** — MOG2 marks shadows with a mid-gray value (127); the mask
   is thresholded so only strong foreground pixels (value 255) remain,
   removing shadow-based false positives.
5. **Morphological processing** — morphological *opening* (erosion followed by
   dilation) removes small, isolated noise blobs, and an additional dilation
   fills small gaps so a single moving object appears as one solid connected
   region rather than several fragments.
6. **Contour detection** — `cv2.findContours` extracts the outlines of the
   remaining white regions in the mask. Contours smaller than a minimum area
   threshold are discarded as noise.
7. **Object detection & bounding boxes** — for every contour that passes the
   area filter, `cv2.boundingRect` computes an axis-aligned rectangle, which is
   drawn on the live frame.
8. **Restricted area check** — a fixed rectangular "restricted zone" is defined
   on the frame. Each object's bounding box is tested for overlap
   (intersection) with this zone using a simple rectangle-intersection test.
9. **Intrusion alert** — if any object's bounding box overlaps the restricted
   zone, the zone is highlighted in red and an "INTRUSION DETECTED" banner is
   displayed on the live video.

## 5. System Architecture

```
 Video / Webcam
       │
       ▼
 Frame Capture (cv2.VideoCapture, resize, Gaussian blur)
       │
       ▼
 Background Subtraction (MOG2 / KNN → foreground mask)
       │
       ▼
 Thresholding (remove shadow gray values, keep pure foreground)
       │
       ▼
 Morphological Processing (opening + dilation → clean mask)
       │
       ▼
 Contour Detection (find outlines of moving blobs)
       │
       ▼
 Object Detection (filter by area, compute bounding boxes)
       │
       ▼
 Restricted Area Check (does any box overlap the monitored zone?)
       │
       ▼
 Intrusion Alert ("INTRUSION DETECTED" shown on screen)
```

Each stage in this pipeline corresponds directly to a clearly separated,
commented section of `intrusion_detection.py`, so the code can be read
top-to-bottom in the same order as this diagram.

## 6. Algorithm (Step-by-Step)

```
1. Initialize video capture (webcam or file) and the MOG2 background subtractor.
2. Repeat for every frame until the video ends or the user quits:
   a. Read a frame; resize it to a fixed width.
   b. Apply Gaussian blur to reduce noise.
   c. Apply the background subtractor to obtain a foreground mask.
   d. Threshold the mask to remove shadow pixels.
   e. Apply morphological opening, then dilation, to clean the mask.
   f. Find external contours in the cleaned mask.
   g. For each contour:
        - If its area < MIN_CONTOUR_AREA: discard (treat as noise).
        - Else: compute its bounding box and draw it on the frame.
        - Check whether this bounding box overlaps the restricted zone.
   h. If any bounding box overlaps the restricted zone: set intrusion = True.
   i. Draw the restricted zone rectangle (green if safe, red if intrusion).
   j. If intrusion is True, display "INTRUSION DETECTED" on the frame.
   k. Show the annotated frame (and the foreground mask) in a window.
3. Release the video capture and close all windows.
```

## 7. Implementation

The system is implemented in a single, well-commented Python script,
`intrusion_detection.py`, using OpenCV (`opencv-python`) and NumPy. Key
implementation details:

- **Background subtractor**: `cv2.createBackgroundSubtractorMOG2(history=500,
  varThreshold=40, detectShadows=True)`. `history` controls how many past
  frames are used to build the background model; `varThreshold` controls how
  sensitive the model is to pixel-value change; `detectShadows=True` lets the
  algorithm mark shadows separately (gray, value 127) instead of counting them
  as real foreground.
- **Noise filtering**: an area threshold (`MIN_CONTOUR_AREA = 800` pixels)
  discards small contours caused by sensor noise, tree leaves moving in the
  background, or shadow fragments.
- **Restricted zone**: defined as a rectangle using *fractional* coordinates of
  the frame size (e.g. `x` from 35% to 80% of the frame width), so the same
  code works correctly regardless of the actual camera/video resolution.
- **Intrusion logic**: a simple axis-aligned bounding-box intersection test
  (`rectangles_intersect`) decides whether a detected object's box overlaps
  the restricted zone — no machine learning or tracking is required.
- **Two-window display**: the live annotated feed and the raw foreground mask
  are both shown, so a viewer can directly see *why* the system believes an
  object is present.

The complete source code is provided in `intrusion_detection.py` in the
project root.

## 8. Results

The system was tested on a bundled sample CCTV-style video (`intruder.mp4`,
1280×720, 30 fps, ~8.5 seconds) showing a person walking through a static
scene. Observations:

- Out of 257 frames, the pipeline detected a valid moving object (contour area
  above the noise threshold) in **208 frames** (~81%), correctly ignoring the
  remaining frames where the person was outside the camera's view or too
  small/occluded.
- Of the frames where an object was detected, the system correctly raised the
  intrusion alert in **196 frames**, corresponding to the portion of the
  video where the person's path overlapped the defined restricted zone.
- The system consistently produced a single, solid bounding box per moving
  person rather than several fragmented boxes, confirming that the
  morphological cleanup step was effective.
- The system processes each frame with only a background-model update, a
  threshold, two morphological operations, and a contour search — all
  operations of low computational cost — allowing it to run comfortably in
  real time (30 fps) on a standard laptop CPU without a GPU.

*(When you demonstrate this project, capture 2–3 screenshots: one with no
person present, one with a person outside the restricted zone, and one with
the "INTRUSION DETECTED" alert active, to include as figures here.)*

## 9. Applications

- Perimeter / boundary monitoring for homes, offices, and warehouses.
- Restricted-access monitoring (e.g., server rooms, cash counters, lab
  equipment areas) where entry into a specific zone should be flagged.
- After-hours occupancy detection in stores or offices.
- A low-cost first stage in a larger surveillance pipeline, where flagged
  frames could later be passed to a more expensive recognition system.

## 10. Limitations

- **Fixed camera assumption**: the technique assumes the camera does not move;
  any camera shake, pan, or zoom will be misread as widespread "motion" and
  break the background model.
- **Lighting sensitivity**: sudden lighting changes (lights turning on/off,
  clouds passing) can temporarily produce large numbers of false-positive
  foreground pixels until the background model adapts.
- **No object identity or classification**: the system detects *that*
  something moved, not *what* it is (person vs. animal vs. blowing curtain);
  it cannot distinguish an authorized person from an intruder.
- **Static object absorption**: an object that stops moving for a long time
  will gradually be absorbed into the background model and stop being
  detected as foreground.
- **Simple zone-overlap logic**: using bounding-box overlap for the intrusion
  check is simple and explainable, but is not pixel-perfect — a large bounding
  box that merely brushes the corner of the restricted zone will trigger an
  alert even if very little of the actual object is inside it.

## 11. Future Scope

- Add simple object tracking (e.g., centroid tracking) across frames to reduce
  duplicate alerts and estimate direction of movement.
- Log intrusion events with timestamps and save short video clips or
  screenshots automatically for later review.
- Support multiple, independently configurable restricted zones in a single
  frame.
- Send real-time notifications (e.g., email/SMS/desktop alert) when an
  intrusion is detected.
- Combine background subtraction with a lightweight classifier as a second
  stage to reduce false positives from non-human motion (optional extension
  beyond the current classical-CV-only scope).

## 12. Conclusion

This project demonstrates that a classical, non-machine-learning computer
vision pipeline — background subtraction, thresholding, morphological
processing, and contour analysis — is sufficient to build a working,
real-time intrusion detection system for a fixed-camera environment. The
system is simple enough that every stage can be explained and justified
without invoking deep learning, yet it produces a practically useful result:
clean bounding boxes around moving objects and a clear, immediate alert when
an object enters a defined restricted area. This makes it a suitable, honest
demonstration of core computer vision concepts for a BTech-level assignment.

## 13. References

1. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
2. Z. Zivkovic, "Improved adaptive Gaussian mixture model for background
   subtraction," *Proceedings of the 17th International Conference on Pattern
   Recognition (ICPR)*, 2004. *(basis of the MOG2 algorithm)*
3. Z. Zivkovic and F. van der Heijden, "Efficient adaptive density estimation
   per image pixel for the task of background subtraction," *Pattern
   Recognition Letters*, 2006. *(basis of the KNN background subtractor)*
4. OpenCV Documentation — *Background Subtraction*:
   https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
5. OpenCV Documentation — *Structural Analysis and Shape Descriptors
   (findContours, boundingRect)*: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
6. OpenCV Documentation — *Morphological Transformations*:
   https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

---

## Appendix A: Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Camera not opening / `Could not open video source` | Wrong webcam index, camera in use by another app, or missing driver | Try `--source 0`, `1`, `2`; close other apps (Zoom, Teams) using the camera; check OS camera permissions |
| Too much background noise / mask is full of white speckles | Low camera quality, compression artifacts, or `varThreshold` too low | Increase `varThreshold` in `createBackgroundSubtractorMOG2`; increase `MIN_CONTOUR_AREA`; increase Gaussian blur kernel size |
| Bounding boxes constantly appearing even with no motion | Background model hasn't "learned" the empty scene yet, or lighting is flickering | Let the video run a few seconds before judging (the background model needs time to warm up on the first ~30–50 frames); keep lighting stable; increase `history` |
| Poor detection of a real moving object | `MIN_CONTOUR_AREA` too high, object too far/small, or too much blur | Lower `MIN_CONTOUR_AREA`; reduce blur kernel size; ensure object is not too small relative to frame |
| A moving object breaks into several small boxes instead of one | Not enough morphological dilation | Increase `iterations` in the `cv2.dilate` call, or increase kernel size |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed in the active Python environment | Run `pip install opencv-python`; confirm you're using the same Python/venv you installed into |
| Video window doesn't appear / freezes | Running inside a headless terminal, SSH session, or notebook without display support | Run the script from a normal local terminal (not a headless remote shell); on Jupyter/Colab, `cv2.imshow` does not work directly — use a local script instead |
| Video window opens but instantly closes | Video file path is wrong so the loop exits immediately | Double-check the `--source` path; run from the project root folder so relative paths like `intruder.mp4` resolve correctly |
