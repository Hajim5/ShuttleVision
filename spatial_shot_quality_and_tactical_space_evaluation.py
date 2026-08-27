import cv2
import numpy as np
import os
from ultralytics import YOLO

H = np.load("homography_matrix.npy")
pose_model = YOLO("yolov8n-pose.pt")
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
shot_quality_distance = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    minimap = np.zeros((court_h, court_w, 3), dtype=np.uint8)
    minimap[:] = (40, 120, 40)
    cv2.line(minimap, (0, net_y), (court_w, net_y), (255, 255, 255), 2)

    # 1. Track Players
    pose_res = pose_model(frame, verbose=False)[0]
    team_top, team_bottom = [], []

    if pose_res.keypoints is not None and len(pose_res.keypoints.data) > 0:
        for kp in pose_res.keypoints.data:
            if kp[15][2] > 0.2 and kp[16][2] > 0.2:
                fx = (kp[15][0].item() + kp[16][0].item()) / 2.0
                fy = (kp[15][1].item() + kp[16][1].item()) / 2.0
                px, py = transform_point(fx, fy, H)

                if -40 <= px <= court_w + 40 and -40 <= py <= court_h + 40:
                    if py < net_y:
                        team_top.append((px, py))
                        cv2.circle(minimap, (px, py), 7, (255, 0, 0), -1)
                    else:
                        team_bottom.append((px, py))
                        cv2.circle(minimap, (px, py), 7, (0, 0, 255), -1)

    # 2. Track Shuttlecock & Evaluate Space
    shuttle_res = shuttle_model(frame, conf=0.25, verbose=False)[0]

    if len(shuttle_res.boxes) > 0:
        box = shuttle_res.boxes[0].xyxy[0].cpu().numpy()
        sx, sy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(frame, (sx, sy), 5, (0, 255, 255), -1)

        s_court_x, s_court_y = transform_point(sx, sy, H)

        if 0 <= s_court_x <= court_w and 0 <= s_court_y <= court_h:
            cv2.circle(minimap, (s_court_x, s_court_y), 6, (0, 255, 255), -1)

            if prev_shuttle_y_court is not None:
                # Crossed bottom to top
                if prev_shuttle_y_court >= net_y and s_court_y < net_y:
                    rally_count += 1
                    if team_top:
                        shot_quality_distance = min([np.linalg.norm(np.array([s_court_x, s_court_y]) - np.array(p)) for p in team_top])
                # Crossed top to bottom
                elif prev_shuttle_y_court <= net_y and s_court_y > net_y:
                    rally_count += 1
                    if team_bottom:
                        shot_quality_distance = min([np.linalg.norm(np.array([s_court_x, s_court_y]) - np.array(p)) for p in team_bottom])

            prev_shuttle_y_court = s_court_y

    # HUD Visuals
    cv2.putText(frame, f"Rally: {rally_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Shot Quality Dist: {shot_quality_distance:.1f}px", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Tactical Space & Quality Evaluation", frame)
    cv2.imshow("2D Minimap", minimap)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
