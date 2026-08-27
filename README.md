# 🏸 ShuttleVision: Badminton Analytics & Shot Quality Evaluator

An end-to-end computer vision pipeline using **OpenCV Homography**, **YOLOv8-pose**, and a custom-trained **YOLOv8 Shuttlecock Detector** to transform standard broadcast footage into top-down 2D court analytics.

---

## 📽️ Project Demos

### 1. Court Homography & Player Tracking
Demonstrates 2D perspective mapping of player foot positions onto the top-down minimap without shuttle tracking.

<!-- Drag and drop your 1st demo video or gif here -->
https://github.com/user-attachments/assets/your-video-id-1.mp4

---

### 2. Shuttlecock Trajectory & Rally Counting
Tracks the high-speed shuttlecock in real time and automatically increments the rally counter upon net crossings.

<!-- Drag and drop your 2nd demo video or gif here -->
"NA"

---

### 3. Shot Placement & Quality Evaluation
Evaluates spatial shot placement by calculating Euclidean distance to defenders on the 2D plane to identify optimal shot execution.

<!-- Drag and drop your 3rd demo video or gif here -->
"NA"

---

## 📌 Features

* **Planar Court Homography:** Calibrates a $3 \times 3$ transformation matrix ($H$) to map broadcast pixel coordinates to a standardized 2D court template.
* **Ground-Contact Player Tracking:** Uses YOLOv8-pose keypoints (ankles) to map court positioning accurately.
* **Custom Shuttlecock Detection:** Lightweight YOLOv8 nano model trained for fast small-object detection.
* **Automated Rally & Shot Analytics:** Tracks net-crossing transitions and computes the distance from the landing point to the nearest opponent:
  $$D = \min\left(\Vert{}P_{\text{shuttle}} - P_{\text{opp1}}\Vert{}, \Vert{}P_{\text{shuttle}} - P_{\text{opp2}}\Vert{}\right)$$

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ShuttleVision.git](https://github.com/your-username/ShuttleVision.git)
   cd ShuttleVision
