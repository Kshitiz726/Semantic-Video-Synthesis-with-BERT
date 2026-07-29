import cv2
import os

video_path = "breaking_bad_clip.mp4"
output_dir = "ppt_frames"
frame_interval = 10   # change if needed

TARGET_W = 1280
TARGET_H = 720

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # resize for PowerPoint
        frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))

        frame_name = f"frame_{saved_count:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame_resized)

        saved_count += 1

    frame_count += 1

cap.release()

print(f"Saved {saved_count} PPT-ready frames in '{output_dir}'")
