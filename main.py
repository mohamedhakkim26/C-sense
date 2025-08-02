from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import shutil
import os
from datetime import datetime

app = FastAPI()
received_alerts = []

# Folder to save snapshots
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

@app.post("/alert")
async def receive_alert(
    label: str = Form(...),
    confidence: float = Form(...),
    timestamp: str = Form(...),
    image: UploadFile = File(...)
):
    # Save image with timestamped name
    image_filename = f"{label}_{timestamp.replace(':','-').replace(' ','_')}.jpg"
    image_path = os.path.join(SNAPSHOT_DIR, image_filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Save alert metadata
    alert = {
        "label": label,
        "confidence": confidence,
        "timestamp": timestamp,
        "image_path": image_path,
        "image_url": f"/snapshots/{image_filename}"
    }
    received_alerts.append(alert)
    print("📨 Alert received:", alert)
    return {"status": "ok"}
@app.get("/alerts", response_class=HTMLResponse)
def show_alerts():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fall Detection Alerts</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f7f7f7;
                margin: 0;
                padding: 20px;
            }
            h2 {
                text-align: center;
                color: #c0392b;
            }
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
            }
            .card img {
                width: 100%;
                height: 180px;
                object-fit: cover;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .meta {
                font-size: 14px;
                color: #333;
            }
        </style>
    </head>
    <body>
        <h2>🚨 Fall Detection Alerts</h2>
        <div class="gallery">
    """
    for alert in received_alerts[::-1]:
        html += f"""
            <div class="card">
                <img src="{alert['image_url']}" alt="Snapshot">
                <div class="meta">
                    <strong>{alert['label'].capitalize()}</strong><br>
                    Confidence: {alert['confidence']}<br>
                    Time: {alert['timestamp']}
                </div>
            </div>
        """
    html += """
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>🚀 Alert receiver running! Visit <a href='/alerts'>/alerts</a> to see all alerts</h2>"

# Serve static snapshot images
from fastapi.staticfiles import StaticFiles
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
