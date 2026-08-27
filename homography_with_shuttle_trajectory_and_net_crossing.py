import cv2
import numpy as np
import os
from ultralytics import YOLO

H = np.load("homography_matrix.npy")
shuttle_model = YOLO("best.pt")

cap = cv2.VideoCapture("rally.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

court_w, court_h = 300, 600
net_y = court_h // 2

def transform_point(x, y, H_matrix):
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H_matrix)
    return int(warped[0][0][0]), int(warped[0][0][1])

rally_count = 0
prev_shuttle_y_court = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    minimap = np.zeros((court_h, court_w, 3), dtype=np.uint8)
    minimap[:] = (40, 120, 40)
    cv2.line(minimap, (0, net_y), (court_w, net_y), (255, 255, 255), 2)

    shuttle_res = shuttle_model(frame, conf=0.25, verbose=False)[0]

    if len(shuttle_res.boxes) > 0:
        box = shuttle_res.boxes[0].xyxy[0].cpu().numpy()
        sx, sy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(frame, (sx, sy), 5, (0, 255, 255), -1)

        s_court_x, s_court_y = transform_point(sx, sy, H)

        if 0 <= s_court_x <= court_w and 0 <= s_court_y <= court_h:
            cv2.circle(minimap, (s_court_x, s_court_y), 6, (0, 255, 255), -1)

            # Net crossing check
            if prev_shuttle_y_court is not None:
                if (prev_shuttle_y_court >= net_y and s_court_y < net_y) or \
                   (prev_shuttle_y_court <= net_y and s_court_y > net_y):
                    rally_count += 1

            prev_shuttle_y_court = s_court_y

    cv2.putText(frame, f"Rally Count: {rally_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Shuttle & Rally Tracker", frame)
    cv2.imshow("2D Minimap", minimap)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
