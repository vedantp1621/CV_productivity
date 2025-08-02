from flask import Blueprint, render_template, request, current_app, jsonify
import os
import threading
import base64
from datetime import datetime

from background.capture_frames import capture_images_every_second
from background.object_detector import yolo_detection

main = Blueprint('main', __name__)

capture_frames_thread = None

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

# Runs capture_frames.py on a new thread
@main.route("/start-monitoring", methods=["POST"])
def start_monitoring():
    global capture_frames_thread
    if capture_frames_thread is None or not capture_frames_thread.is_alive():
        capture_frames_thread = threading.Thread(
            target=capture_images_every_second, daemon=True
        )
        capture_frames_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})


@main.route("/test", methods=["POST"])
def test():
    yolo_detection()
    return "Yolo running!"
