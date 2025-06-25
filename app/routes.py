from flask import render_template, request, redirect
import os
import threading

from background.tracker import start_tracking
from model.trainer import train_model
from model.object_detector import yolo_detection


def init_routes(app):

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload_image():
        image = request.files.get("image")
        if image:
            save_path = os.path.join("training_data", image.filename)
            image.save(save_path)
        return redirect("/")

    @app.route("/train", methods=["POST"])
    def train():
        train_model()
        return "Model trained!"

    @app.route("/start", methods=["POST"])
    def start():
        thread = threading.Thread(target=start_tracking)
        thread.daemon = True
        thread.start()
        return "Monitoring started!"

    @app.route("/test", methods=["POST"])
    def test():
        yolo_detection()
        return "Yolo running!"
