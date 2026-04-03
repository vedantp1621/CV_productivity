import os
import time
import threading
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Blueprint, render_template, request, current_app, jsonify, Response

from background.capture import capture_loop
from background.phone_detector import phone_loop
from background.attention_detector import attention_loop
from background.shared import (
    monitoring_active, detection_state, detection_lock,
    session_stats, session_lock,
    latest_annotated, annotated_lock,
)

main = Blueprint('main', __name__)

_capture_thread   = None
_phone_thread     = None
_attention_thread = None


def _idle_jpeg():
    """Static placeholder frame served when monitoring is inactive."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Start monitoring to see the live feed",
                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


_IDLE_FRAME = None   # generated once on first request


def _gen_frames():
    global _IDLE_FRAME
    while True:
        with annotated_lock:
            jpeg = latest_annotated.get("jpeg")
        if jpeg:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        else:
            if _IDLE_FRAME is None:
                _IDLE_FRAME = _idle_jpeg()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _IDLE_FRAME + b'\r\n'
        time.sleep(0.033)   # ~30 fps cap


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/start-monitoring", methods=["POST"])
def start_monitoring():
    global _capture_thread, _phone_thread, _attention_thread

    if monitoring_active.is_set():
        return jsonify({"status": "already running"})

    monitoring_active.set()

    with session_lock:
        if session_stats["start_time"] is None:   # first start of this session
            session_stats["start_time"]         = time.monotonic()
            session_stats["phone_seconds"]      = 0.0
            session_stats["distracted_seconds"] = 0.0

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


@main.route("/end-session", methods=["POST"])
def end_session():
    monitoring_active.clear()

    with session_lock:
        start        = session_stats.get("start_time")
        phone_s      = session_stats["phone_seconds"]
        distracted_s = session_stats["distracted_seconds"]
        # Reset so the next session starts fresh
        session_stats["start_time"]         = None
        session_stats["phone_seconds"]      = 0.0
        session_stats["distracted_seconds"] = 0.0

    total_s   = (time.monotonic() - start) if start else 0.0
    focused_s = max(0.0, total_s - distracted_s)

    return jsonify({
        "total_seconds":      round(total_s),
        "phone_seconds":      round(phone_s),
        "distracted_seconds": round(distracted_s),
        "focused_seconds":    round(focused_s),
    })


@main.route("/video-feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@main.route("/status")
def status():
    with detection_lock:
        return jsonify(dict(detection_state))
