import os
import cv2
import csv
import time
import math
import torch
import requests
from ultralytics import YOLO

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8-pose model to device
model = YOLO("yolov8m-pose.pt").to(device)

def compute_shoulder_hip_angle(keypoints, confidences):
    try:
        # Left shoulder = 5, Left hip = 11 (based on COCO keypoints)
        if confidences[5] < 0.5 or confidences[11] < 0.5:
            return None
        dx = keypoints[11][0] - keypoints[5][0]
        dy = keypoints[11][1] - keypoints[5][1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = abs(math.degrees(angle_rad))
        return angle_deg
    except Exception as e:
        print(f"⚠ Error in angle calculation: {e}")
        return None

def send_alert(callback_url, frame_number, angle_logged, image_path):
    try:
        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            data = {
                "label": "fall_detected",
                "confidence": 1.0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_number": frame_number,
                "angle": angle_logged
            }
            response = requests.post(callback_url, data=data, files=files)
            print(f"📤 Alert sent | Response: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to send alert: {e}")

def run_fall_detection(rtsp_url, callback_url, stop_event):
    print(f"🔍 Starting fall detection from: {rtsp_url}")

    save_dir = "fall_frames"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"❌ Cannot open video stream: {rtsp_url}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2) if fps > 0 else 60  # Process every 2 seconds
    frame_count = 0

    csv_path = "alerts.csv"
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["frame", "person_id", "angle", "status"])

    while cap.isOpened() and not stop_event.is_set():
        ret = cap.grab()
        if not ret:
            print("⚠ Failed to grab frame. Exiting...")
            break

        if frame_count % frame_interval == 0:
            success, frame = cap.retrieve()
            if not success:
                print("⚠ Failed to retrieve frame.")
                frame_count += 1
                continue

            try:
                results = model.predict(frame, conf=0.25, save=False, imgsz=960, device=device)
                if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                    print("⚠ No keypoints detected.")
                    frame_count += 1
                    continue
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                frame_count += 1
                continue

            output_frame = frame.copy()
            alert_this_frame = False
            angle_logged = {}

            keypoints_all = results[0].keypoints.xy
            confidences_all = results[0].keypoints.conf
            classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes else []

            for i, keypoints_tensor in enumerate(keypoints_all):
                if len(classes) > i and int(classes[i]) != 0:  # 0 = person
                    continue

                keypoints = keypoints_tensor.detach().cpu().numpy()
                confidences = confidences_all[i].detach().cpu().numpy() if confidences_all is not None else None
                if confidences is None:
                    continue

                angle = compute_shoulder_hip_angle(keypoints, confidences)
                alert = "No Alert"

                if angle is not None:
                    x, y = int(keypoints[5][0]), int(keypoints[5][1])
                    cv2.putText(output_frame, f"Angle: {angle:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if angle < 40 or angle > 140:
                        alert = "Alert"
                        alert_this_frame = True
                        cv2.putText(output_frame, "⚠ FALL", (x, y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        angle_logged[i] = round(angle, 2)

                csv_writer.writerow([frame_count, i, f"{angle:.2f}" if angle else "N/A", alert])

            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.detach().cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if alert_this_frame:
                filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(filename, output_frame)
                send_alert(callback_url, frame_count, angle_logged, filename)

        frame_count += 1
        time.sleep(0.01)  # minimal sleep

    cap.release()
    csv_file.close()
    print("✅ Fall detection finished.")
