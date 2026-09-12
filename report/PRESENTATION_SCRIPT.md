# 1-Minute Presentation Script

Natural spoken script (~140 words, ~1 minute at a normal pace). Practice it a
couple of times so it sounds like you explaining it, not reading it.

---

"Hi, I'm presenting my project: Smart Surveillance for Intrusion Detection in
a Fixed Environment using Background Subtraction.

The idea is simple — most CCTV cameras are static, so anything that changes
in the frame is very likely a moving object. My system uses OpenCV's MOG2
background subtraction algorithm to build a model of the static background
and separate out anything moving in front of it.

Once I have that foreground mask, I clean it up using thresholding and
morphological operations to remove noise, then use contour detection to find
the actual shape of each moving object and draw a bounding box around it.

I've also marked a restricted zone on the screen. When a moving object's
bounding box enters that zone, the system immediately displays 'INTRUSION
DETECTED' on screen, in real time.

Let me show you a quick live demo... [walk into frame, then into the marked
zone] ...as you can see, the alert triggers the moment I step into the
restricted area.

This approach is lightweight, explainable, and doesn't need any deep learning
— making it practical for real fixed-camera security setups like server rooms
or store entrances."

---

## Demo Blocking Notes (what to actually do on camera)

1. Start the script with nobody in frame — let it run 3-5 seconds so the
   background model "settles" (point this out: "no alert, no boxes, because
   nothing is moving").
2. Walk into frame but stay outside the restricted zone — point out the green
   bounding box following you, and that the zone rectangle stays green.
3. Walk into the restricted zone — the box turns red, the zone rectangle turns
   red, and "INTRUSION DETECTED" appears at the top of the frame.
4. Optionally step back out to show the alert disappears once you leave the
   zone.
