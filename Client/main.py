import os
import cv2
import csv
import time
import math
import torch
import requests
from threading import Event, Thread
from fastapi import FastAPI

from ultralytics import YOLO

app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8m-pose.pt").to(device)

active_threads = []  # [(thread, stop_event)]


def compute_shoulder_hip_angle(keypoints, confidences):
    try:
        if confidences[5] < 0.5 or confidences[11] < 0.5:
            return None
        dx = keypoints[11][0] - keypoints[5][0]
        dy = keypoints[11][1] - keypoints[5][1]
        angle_rad = torch.atan2(dy, dx)
        return abs(torch.rad2deg(angle_rad).item())
    except Exception as e:
        print(f"⚠ Error in angle calc: {e}")
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


def run_fall_detection(http_url, callback_url, stop_event):
    print(f"🔍 Starting fall detection from HTTP stream: {http_url}")

    save_dir = "fall_frames"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(http_url)
    if not cap.isOpened():
        print(f"❌ Cannot open HTTP video stream: {http_url}")
        return

    last_detected_time = 0
    frame_count = 0

    csv_path = "alerts.csv"
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(["frame", "person_id", "angle", "status"])

        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠ Failed to read frame. Exiting...")
                break

            now = time.time()
            if now - last_detected_time < 3:
                time.sleep(0.1)
                continue

            last_detected_time = now

            try:
                results = model.predict(frame, conf=0.25, save=False, imgsz=960, device=device)
                if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                    continue
            except Exception as e:
                print(f"❌ Model error: {e}")
                continue

            output_frame = frame.copy()
            alert_this_frame = False
            angle_logged = {}

            keypoints_all = results[0].keypoints.xy.to(device)
            confidences_all = results[0].keypoints.conf.to(device) if results[0].keypoints.conf is not None else None
            classes = results[0].boxes.cls if results[0].boxes else torch.tensor([]).to(device)

            for i, keypoints in enumerate(keypoints_all):
                if len(classes) > i and int(classes[i].item()) != 0:
                    continue
                confidences = confidences_all[i] if confidences_all is not None else None
                if confidences is None:
                    continue

                angle = compute_shoulder_hip_angle(keypoints, confidences)
                alert = "No Alert"

                if angle is not None:
                    x, y = int(keypoints[5][0].item()), int(keypoints[5][1].item())
                    if angle < 40 or angle > 140:
                        alert = "Alert"
                        alert_this_frame = True
                        angle_logged[i] = round(angle, 2)
                        cv2.putText(output_frame, "⚠ FALL", (x, y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                csv_writer.writerow([frame_count, i, f"{angle:.2f}" if angle else "N/A", alert])

            if alert_this_frame:
                filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(filename, output_frame)
                send_alert(callback_url, frame_count, angle_logged, filename)

            frame_count += 1

        cap.release()
        print("✅ Fall detection finished.")


@app.get("/process-stream")
def process_stream(http_url: str, callback_url: str):
    stop_event = Event()

    def detection_runner():
        run_fall_detection(http_url, callback_url, stop_event)

    thread = Thread(target=detection_runner, daemon=True)
    thread.start()
    active_threads.append((thread, stop_event))
    return {"status": "Fall detection started", "source": http_url}


@app.get("/stop-stream")
def stop_stream():
    for thread, stop_event in active_threads:
        stop_event.set()
    active_threads.clear()
    return {"status": "All detection threads stopped"}
