# main.py

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from threading import Thread, Event
import shutil, os
from datetime import datetime
from alpr_engine import run_alpr_stream

app = FastAPI()
received_alerts = []
active_threads = []

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Serve image snapshots
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

@app.post("/alert")
async def receive_alert(
    label: str = Form(...),
    plate: str = Form(...),
    confidence: float = Form(...),
    timestamp: str = Form(...),
    image: UploadFile = File(...)
):
    image_filename = f"{label}_{plate}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
    image_path = os.path.join(SNAPSHOT_DIR, image_filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    alert = {
        "label": label,
        "plate": plate,
        "confidence": confidence,
        "timestamp": timestamp,
        "image_path": f"snapshots/{image_filename}"
    }
    received_alerts.append(alert)
    print("📨 ALPR Alert received:", alert)
    return {"status": "ok", "plate": plate}

@app.get("/alerts")
def get_alerts():
    return {"alerts": received_alerts}

@app.get("/process-stream")
def process_stream(http_url: str = Query(...), callback_url: str = Query(...)):
    stop_event = Event()
    thread = Thread(target=run_alpr_stream, args=(http_url, callback_url, stop_event), daemon=True)
    thread.start()
    active_threads.append((thread, stop_event))
    return {"status": "ALPR stream started"}

@app.get("/stop-stream")
def stop_stream():
    for _, stop_event in active_threads:
        stop_event.set()
    active_threads.clear()
    return {"status": "All streams stopped"}

@app.get("/", response_class=HTMLResponse)
def root():
    html = "<h2>🚘 License Plate Alerts</h2><div style='display: flex; flex-wrap: wrap;'>"
    for alert in reversed(received_alerts):
        html += f"""
        <div style='margin: 10px; border: 1px solid #ccc; padding: 10px; width: 240px'>
            <img src='/{alert["image_path"]}' width='220' /><br>
            <strong>Plate:</strong> {alert['plate']}<br>
            <strong>Confidence:</strong> {alert['confidence']:.2f}<br>
            <small>{alert['timestamp']}</small>
        </div>
        """
    html += "</div>"
    return HTMLResponse(content=html)
