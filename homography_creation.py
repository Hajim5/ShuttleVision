import cv2
import numpy as np
import os

video_path = "rally.mp4"
if not os.path.exists(video_path):
    print(f"Error: '{video_path}' not found.")
    exit()

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Could not read frame from video.")

src_pts = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        src_pts.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(len(src_pts)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Select 4 Corners (TL, TR, BR, BL)", frame)

print("Click 4 outer court corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
cv2.imshow("Select 4 Corners (TL, TR, BR, BL)", frame)
cv2.setMouseCallback("Select 4 Corners (TL, TR, BR, BL)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(src_pts) == 4:
    court_w, court_h = 300, 600
    dst_pts = np.float32([
        [0, 0],
        [court_w, 0],
        [court_w, court_h],
        [0, court_h]
    ])

    src_arr = np.float32(src_pts)
    H, _ = cv2.findHomography(src_arr, dst_pts)

    np.save("homography_matrix.npy", H)
    print("Homography Matrix successfully computed and saved to 'homography_matrix.npy'!")
else:
    print("Error: You must select exactly 4 points.")
