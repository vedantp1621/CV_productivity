import cv2
import numpy as np
import os
import threading
import base64
from datetime import datetime

from flask import Blueprint, render_template, request, current_app, jsonify

from background.capture import capture_loop
from background.unified_detector import unified_loop
from background.shared import (
    monitoring_active,
    detection_state,
    detection_lock,
    config_complete,
    reference_embeddings,
)
from model.embedder import embed_image

main = Blueprint('main', __name__)

_capture_thread = None
_detector_thread = None


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


@main.route("/configure", methods=["POST"])
def configure():
    phone_keys = ["phone_front", "phone_back"]
    pose_keys = [f"pose_{i}" for i in range(1, 6)]

    phone_embeddings = []
    pose_embeddings = []

    for key in phone_keys:
        f = request.files.get(key)
        if f is None:
            return jsonify({"error": f"Missing file: {key}"}), 400
        data = np.frombuffer(f.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": f"Could not decode image: {key}"}), 400
        phone_embeddings.append(embed_image(frame))

    for key in pose_keys:
        f = request.files.get(key)
        if f is None:
            return jsonify({"error": f"Missing file: {key}"}), 400
        data = np.frombuffer(f.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": f"Could not decode image: {key}"}), 400
        pose_embeddings.append(embed_image(frame))

    reference_embeddings["phone"] = phone_embeddings
    reference_embeddings["pose"] = pose_embeddings
    config_complete.set()

    return jsonify({"status": "ok", "embeddings_computed": len(phone_embeddings) + len(pose_embeddings)})


@main.route("/start-monitoring", methods=["POST"])
def start_monitoring():
    global _capture_thread, _detector_thread

    if not config_complete.is_set():
        return jsonify({"error": "Run /configure first"}), 400

    if monitoring_active.is_set():
        return jsonify({"status": "already running"})

    monitoring_active.set()

    webcam_index = current_app.config.get("WEBCAM_INDEX", 0)

    _capture_thread = threading.Thread(
        target=capture_loop, args=(webcam_index,), daemon=True
    )
    _detector_thread = threading.Thread(target=unified_loop, daemon=True)

    _capture_thread.start()
    _detector_thread.start()

    return jsonify({"status": "started"})


@main.route("/status")
def status():
    with detection_lock:
        return jsonify(dict(detection_state))


@main.route("/stop-monitoring", methods=["POST"])
def stop_monitoring():
    monitoring_active.clear()
    return jsonify({"status": "stopped"})
