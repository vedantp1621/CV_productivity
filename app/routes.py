import os
import threading
import base64
from datetime import datetime

from flask import Blueprint, render_template, request, current_app, jsonify

from background.capture import capture_loop
from background.phone_detector import phone_loop
from background.attention_detector import attention_loop
from background.shared import monitoring_active, detection_state, detection_lock

main = Blueprint('main', __name__)

_capture_thread  = None
_phone_thread    = None
_attention_thread = None


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/start-monitoring", methods=["POST"])
def start_monitoring():
    global _capture_thread, _phone_thread, _attention_thread

    if monitoring_active.is_set():
        return jsonify({"status": "already running"})

    monitoring_active.set()

    webcam_index = current_app.config.get("WEBCAM_INDEX", 0)

    _capture_thread   = threading.Thread(target=capture_loop, args=(webcam_index,), daemon=True)
    _phone_thread     = threading.Thread(target=phone_loop,   daemon=True)
    _attention_thread = threading.Thread(target=attention_loop, daemon=True)

    _capture_thread.start()
    _phone_thread.start()
    _attention_thread.start()

    return jsonify({"status": "started"})


@main.route("/stop-monitoring", methods=["POST"])
def stop_monitoring():
    monitoring_active.clear()
    return jsonify({"status": "stopped"})


@main.route("/status")
def status():
    with detection_lock:
        return jsonify(dict(detection_state))
