from flask import Blueprint, render_template, request, current_app, jsonify
import os
import threading
import base64
from datetime import datetime

from background.capture import capture_loop
from background.predictor import yolo_loop
from background.gaze import gaze_loop
from background.shared import monitoring_active, detection_state, detection_lock

main = Blueprint('main', __name__)

_capture_thread = None
_yolo_thread = None
_gaze_thread = None


@main.route("/")
def home():
    folder = current_app.config["UPLOAD_FOLDER"]
    image_count = len(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    return render_template("index.html", image_count=image_count)


# Saves images uploaded by user
@main.route("/capture", methods=["POST"])
def capture():
    data_url = request.form["image"]
    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)

    folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(binary_data)

    image_count = len(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    return jsonify({"message": f"Saved: {filename}", "count": image_count})


@main.route("/start-monitoring", methods=["POST"])
def start_monitoring():
    global _capture_thread, _yolo_thread, _gaze_thread

    if monitoring_active.is_set():
        return jsonify({"status": "already running"})

    monitoring_active.set()

    webcam_index = current_app.config.get("WEBCAM_INDEX", 0)

    _capture_thread = threading.Thread(
        target=capture_loop, args=(webcam_index,), daemon=True
    )
    _yolo_thread = threading.Thread(target=yolo_loop, daemon=True)
    _gaze_thread = threading.Thread(target=gaze_loop, daemon=True)

    _capture_thread.start()
    _yolo_thread.start()
    _gaze_thread.start()

    return jsonify({"status": "started"})


@main.route("/status")
def status():
    with detection_lock:
        return jsonify(dict(detection_state))


@main.route("/stop-monitoring", methods=["POST"])
def stop_monitoring():
    monitoring_active.clear()
    return jsonify({"status": "stopped"})


@main.route("/test", methods=["POST"])
def test():
    return jsonify({
        "status": "YOLO detection is now part of the monitoring pipeline.",
        "hint": "POST to /start-monitoring to begin real-time detection."
    })
