**Label Explanation**
B1: Out of hole
B2: Proximity to ROI
B3: Out of body
C1: Image blur
C2: Improper exposure
D1: Smoke
D2: Bleeding
For example: "video01_005_B1_C2_D1_B3.mp4" indicates there are Out of hole, Improper exposure, Smoke
 and Out of body in this video.

---
Each row in annotations.txt corresponds to a video and contains one or more annotated intraoperative adverse event (iAE) intervals.

For example:

  **video52_002_C2_label_C2-0-0-0 320 351 400 511**

 -The first column is the video name.

 - The following numbers are start-end frame pairs indicating time intervals in which iAEs occur.

 - In this example, two iAE segments are annotated:

 - - From frame 320 to 351,

 - - From frame 400 to 511.
