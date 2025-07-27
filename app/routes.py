from flask import Blueprint, render_template, request, current_app
import os
import threading
import base64
from datetime import datetime

from background.tracker import start_tracking
from model.trainer import train_model
from model.object_detector import yolo_detection

main = Blueprint('main', __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/capture", methods=["POST"])
def capture():
    data_url = request.form["image"]
    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)

    folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(binary_data)

    return "Saved: " + filename

@main.route("/start", methods=["POST"])
def start():
    thread = threading.Thread(target=start_tracking)
    thread.daemon = True
    thread.start()
    return "Monitoring started!"

@main.route("/test", methods=["POST"])
def test():
    yolo_detection()
    return "Yolo running!"
