import cv2
import numpy as np
import os
from ultralytics import YOLO

if not os.path.exists("homography_matrix.npy"):
    print("Error: Run 'homography_creation.py' first!")
    exit()

H = np.load("homography_matrix.npy")
pose_model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture("rally.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

court_w, court_h = 300, 600
net_y = court_h // 2

def transform_point(x, y, H_matrix):
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H_matrix)
    return int(warped[0][0][0]), int(warped[0][0][1])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    minimap = np.zeros((court_h, court_w, 3), dtype=np.uint8)
    minimap[:] = (40, 120, 40)
    cv2.line(minimap, (0, net_y), (court_w, net_y), (255, 255, 255), 2)

    results = pose_model(frame, verbose=False)[0]

    if results.keypoints is not None and len(results.keypoints.data) > 0:
        for kp in results.keypoints.data:
            l_conf, r_conf = kp[15][2].item(), kp[16][2].item()
            if l_conf > 0.2 and r_conf > 0.2:
                foot_x = (kp[15][0].item() + kp[16][0].item()) / 2.0
                foot_y = (kp[15][1].item() + kp[16][1].item()) / 2.0
            elif l_conf > 0.2:
                foot_x, foot_y = kp[15][0].item(), kp[15][1].item()
            elif r_conf > 0.2:
                foot_x, foot_y = kp[16][0].item(), kp[16][1].item()
            else:
                continue

            px, py = transform_point(foot_x, foot_y, H)

            if -40 <= px <= court_w + 40 and -40 <= py <= court_h + 40:
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
                color = (255, 0, 0) if py < net_y else (0, 0, 255)
                cv2.circle(minimap, (px, py), 7, color, -1)

    cv2.imshow("Player Mapping Frame", frame)
    cv2.imshow("2D Minimap", minimap)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
