"""
Smart Surveillance for Intrusion Detection in a Fixed Environment
using Background Subtraction (OpenCV MOG2)

Pipeline implemented in this file:

    Video/Webcam -> Frame Capture -> Background Subtraction -> Thresholding
    -> Morphological Processing -> Contour Detection -> Object Detection
    -> Restricted Area Check -> Intrusion Alert

Run:
    python intrusion_detection.py                  # uses the bundled sample video
    python intrusion_detection.py --source 0        # uses the default webcam
"""

import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. CONFIGURATION  (the only section a student normally needs to tweak)
# ---------------------------------------------------------------------------

FRAME_WIDTH = 640          # every frame is resized to this width for speed
MIN_CONTOUR_AREA = 800     # contours smaller than this (in pixels) are treated
                            # as noise, not as a real moving object

# Restricted / monitored zone, expressed as a fraction of the frame size so it
# scales automatically to whatever video/webcam resolution is used.
ZONE_X1, ZONE_Y1 = 0.35, 0.35   # top-left corner   (fraction of width, height)
ZONE_X2, ZONE_Y2 = 0.80, 0.90   # bottom-right corner

COLOR_SAFE = (0, 200, 0)       # green  -> no intrusion
COLOR_ALERT = (0, 0, 255)      # red    -> intrusion detected
COLOR_BOX = (255, 200, 0)      # cyan-ish -> bounding box around moving object


def get_restricted_zone(frame_width, frame_height):
    """Convert the fractional zone coordinates into actual pixel coordinates."""
    x1 = int(frame_width * ZONE_X1)
    y1 = int(frame_height * ZONE_Y1)
    x2 = int(frame_width * ZONE_X2)
    y2 = int(frame_height * ZONE_Y2)
    return x1, y1, x2, y2


def rectangles_intersect(rect_a, rect_b):
    """
    Simple axis-aligned bounding box (AABB) overlap test.
    Each rectangle is given as (x1, y1, x2, y2).
    Returns True if the two rectangles overlap at all.
    """
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    if ax2 < bx1 or bx2 < ax1:
        return False
    if ay2 < by1 or by2 < ay1:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="OpenCV background-subtraction intrusion detector")
    parser.add_argument(
        "--source",
        default="intruder.mp4",
        help="Path to a video file, OR a webcam index such as 0 (default: intruder.mp4)",
    )
    parser.add_argument(
        "--algo",
        default="MOG2",
        choices=["MOG2", "KNN"],
        help="Background subtraction algorithm to use (default: MOG2)",
    )
    args = parser.parse_args()

    # A webcam index is passed as a number (e.g. "0"); a video file is a path.
    source = int(args.source) if args.source.isdigit() else args.source

    # -----------------------------------------------------------------------
    # 2. VIDEO / WEBCAM CAPTURE
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        print("        - If using a webcam, check the index (0, 1, ...) and that no other app is using it.")
        print("        - If using a file, check that the path is correct.")
        return

    # -----------------------------------------------------------------------
    # 3. BACKGROUND SUBTRACTOR
    #    MOG2 builds a statistical model of the static background from the
    #    last N frames and labels any pixel that no longer fits that model
    #    as "foreground" (i.e. something moving).
    # -----------------------------------------------------------------------
    if args.algo == "MOG2":
        back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
    else:
        back_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("[INFO] Press 'q' or ESC in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        # Resize every frame to a fixed width so processing speed and the
        # restricted-zone coordinates stay consistent regardless of source.
        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / w
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
        frame_h, frame_w = frame.shape[:2]

        # Light blur to reduce camera/sensor noise before subtracting the
        # background -- this avoids single noisy pixels being picked up as
        # "motion".
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # ---------------------------------------------------------------
        # 4. BACKGROUND SUBTRACTION -> raw foreground mask
        # ---------------------------------------------------------------
        fg_mask = back_sub.apply(blurred)

        # ---------------------------------------------------------------
        # 5. THRESHOLDING
        #    MOG2/KNN mark shadows with the gray value 127. We threshold the
        #    mask so that only strong foreground pixels (value 255) remain,
        #    which removes shadow-related false detections.
        # ---------------------------------------------------------------
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # ---------------------------------------------------------------
        # 6. MORPHOLOGICAL PROCESSING
        #    Opening (erode -> dilate) removes small isolated noise blobs.
        #    A further dilation reconnects/fills the silhouette of the
        #    moving object so it forms one solid contour instead of many
        #    small fragments.
        # ---------------------------------------------------------------
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # ---------------------------------------------------------------
        # 7. CONTOUR DETECTION
        # ---------------------------------------------------------------
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ---------------------------------------------------------------
        # 8. RESTRICTED AREA DEFINITION
        # ---------------------------------------------------------------
        zone = get_restricted_zone(frame_w, frame_h)
        zx1, zy1, zx2, zy2 = zone

        intrusion = False

        # ---------------------------------------------------------------
        # 9. OBJECT DETECTION + RESTRICTED AREA CHECK
        # ---------------------------------------------------------------
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue  # ignore tiny blobs (noise, leaves, compression artifacts)

            x, y, bw, bh = cv2.boundingRect(contour)
            object_rect = (x, y, x + bw, y + bh)

            # Has this moving object entered the restricted zone?
            if rectangles_intersect(object_rect, zone):
                intrusion = True
                box_color = COLOR_ALERT
            else:
                box_color = COLOR_BOX

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)

        # Draw the restricted zone itself: red when breached, green otherwise
        zone_color = COLOR_ALERT if intrusion else COLOR_SAFE
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_color, 2)
        cv2.putText(frame, "Restricted Area", (zx1, max(zy1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

        # ---------------------------------------------------------------
        # 10. INTRUSION ALERT
        # ---------------------------------------------------------------
        if intrusion:
            cv2.rectangle(frame, (0, 0), (frame_w, 40), COLOR_ALERT, -1)
            cv2.putText(frame, "INTRUSION DETECTED", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ---------------------------------------------------------------
        # 11. DISPLAY OUTPUT
        # ---------------------------------------------------------------
        cv2.imshow("Live Feed - Intrusion Detection", frame)
        cv2.imshow("Foreground Mask (Background Subtraction Output)", fg_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
