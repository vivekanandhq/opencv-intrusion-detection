# Viva / Oral Exam Preparation

Short, simple answers you can say out loud without reading. Expand in your own
words if the professor asks a follow-up.

**1. What is background subtraction?**
It's a technique that separates moving "foreground" objects from a static
"background" by comparing the current frame to a model of what the background
normally looks like. Any pixel that changed a lot is labeled foreground.

**2. Why did you choose background subtraction instead of a deep learning
detector like YOLO?**
Because the camera and scene are fixed — the background never changes. In
that situation, a statistical model of "what normally doesn't move" is enough
to find intruders, without needing labeled training data, a GPU, or a large
model. It's simpler, faster, and fully explainable.

**3. What is MOG2?**
MOG2 stands for "Mixture of Gaussians (v2)". It models each pixel's recent
history of values as a mixture of a few Gaussian distributions. If a new
pixel value doesn't fit any of those distributions well, it's classified as
foreground (moving object) rather than background.

**4. How is MOG2 different from KNN background subtraction?**
Both build a per-pixel background model, but MOG2 models each pixel with a
mixture of Gaussian distributions, while KNN (K-Nearest Neighbours) compares
the current pixel value to its K nearest historical samples. KNN can adapt
slightly better to complex/noisy backgrounds; MOG2 is faster and simpler.
Both are provided in this project (`--algo MOG2` or `--algo KNN`).

**5. What do "foreground" and "background" mean here?**
Background = anything that stays the same across many frames (walls, floor,
furniture). Foreground = pixels that changed significantly compared to the
learned background — usually because something is moving there.

**6. Why do you apply thresholding after background subtraction?**
MOG2/KNN mark shadows as gray (value 127), not black (0) or white (255). To
avoid treating shadows as real objects, I threshold the mask so only strong,
pure-white foreground pixels (255) remain.

**7. Why do you need morphological operations? What do they do here?**
The raw foreground mask is noisy — small speckles from sensor noise, and
sometimes a single object appears as several disconnected white blobs. I use
morphological "opening" (erosion followed by dilation) to remove tiny noise
blobs, and additional dilation to fill gaps so one object forms one solid
region instead of many fragments.

**8. What is a contour, and why do you use `findContours`?**
A contour is the outline/boundary of a connected white region in the mask.
`findContours` locates these outlines so I can measure each blob's size and
compute a bounding rectangle around it — this is how individual moving
objects are located in the frame.

**9. Why do you filter contours by area?**
Very small contours are usually noise (a few stray pixels), not real objects.
By ignoring contours smaller than a minimum area (`MIN_CONTOUR_AREA`), the
system reports only meaningfully sized moving objects.

**10. How do you draw the bounding box?**
`cv2.boundingRect(contour)` returns the smallest upright rectangle (x, y,
width, height) that contains the contour. I draw that rectangle on the frame
with `cv2.rectangle`.

**11. What do you mean by a "fixed environment", and why does it matter for
this technique?**
A fixed environment means the camera is mounted in place and does not pan,
tilt, zoom, or move — like a CCTV camera. Background subtraction relies on the
background staying visually consistent frame-to-frame; if the camera moved,
the entire frame would appear to change, and the algorithm would (wrongly)
treat the whole scene as foreground.

**12. How do you detect an "intrusion" specifically?**
I define a rectangular restricted zone on the frame. For every detected
object's bounding box, I check whether it geometrically overlaps that zone
using a simple rectangle-intersection test. If any object's box overlaps the
zone, I flag that frame as an intrusion and display "INTRUSION DETECTED".

**13. What are the limitations of your system?**
It cannot tell *what* the object is (person, animal, object) — only that
something moved. It assumes a fixed camera and relatively stable lighting.
Sudden light changes or camera shake cause false positives. An object that
stays still for a long time gets absorbed into the background and stops being
detected.

**14. What causes false positives in your system, and how did you reduce
them?**
Sensor noise, compression artifacts, shadows, and lighting flicker. I reduced
these using: a Gaussian blur before subtraction, shadow removal via
thresholding, morphological opening to remove small noise blobs, and a
minimum contour area filter.

**15. Can this system work in real time? How fast is it?**
Yes — every stage (background subtraction, threshold, two morphological
operations, contour search) is computationally cheap. It comfortably keeps up
with a 30 fps video/webcam feed on a normal laptop CPU, with no GPU needed.

**16. Why did you resize the frame at the start?**
To keep processing speed consistent regardless of the input resolution, and
so the restricted zone (defined as a fraction of the frame size) lines up
correctly no matter what camera or video is used.

**17. What would you improve if you had more time?**
Add simple centroid-based tracking to avoid duplicate/flickering alerts,
automatically log intrusion events with timestamps and save clips, and
support multiple restricted zones.
