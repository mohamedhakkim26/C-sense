# alpr_engine.py

import os
import cv2
import time
import requests
from datetime import datetime
from threading import Event
from fast_alpr import ALPR

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

DETECTION_INTERVAL = 2  # seconds
DUPLICATE_TIMEOUT = 10  # seconds

def run_alpr_stream(http_url: str, callback_url: str, stop_event: Event):
    print(f"🔍 Starting ALPR on stream: {http_url}")
    cap = cv2.VideoCapture(http_url)
    if not cap.isOpened():
        print(f"❌ Failed to open stream: {http_url}")
        return

    last_detection_time = 0
    seen_plates = {}

    while not stop_event.is_set() and cap.isOpened():
        current_time = time.time()
        if current_time - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read failed. Retrying...")
            time.sleep(1)
            continue

        try:
            results = alpr.predict(frame)
            for result in results:
                plate = result.ocr.text.strip()
                conf = result.ocr.confidence
                now = time.time()

                # Avoid duplicate alerts for same plate
                if plate in seen_plates and now - seen_plates[plate] < DUPLICATE_TIMEOUT:
                    continue

                seen_plates[plate] = now
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename = f"plate_{plate}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
                image_path = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(image_path, frame)

                with open(image_path, "rb") as img_file:
                    files = {"image": img_file}
                    data = {
                        "label": "plate_detected",
                        "plate": plate,
                        "confidence": conf,
                        "timestamp": timestamp
                    }
                    response = requests.post(callback_url, data=data, files=files)
                    print(f"📤 Sent alert for plate: {plate} | Status: {response.status_code}")

        except Exception as e:
            print(f"❌ ALPR error: {e}")

        last_detection_time = current_time

    cap.release()
    print("🛑 Stream closed.")
