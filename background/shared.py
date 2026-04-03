import queue
import threading

# One queue per consumer so every frame reaches both pipelines
yolo_queue      = queue.Queue(maxsize=10)
attention_queue = queue.Queue(maxsize=10)

# Set to start monitoring, clear to stop all loops
monitoring_active = threading.Event()

# Latest detection results — written by consumer threads, read by /status
detection_state = {
    "phone":       False,
    "eyes_off":    False,
    "calibrating": False,
    "delta_pitch": 0.0,    # degrees from baseline (attention_loop)
    "delta_yaw":   0.0,
    "score":       0,      # temporal smoothing score (0–SCORE_MAX)
}
detection_lock = threading.Lock()

# Annotated JPEG frame — written by attention_loop, streamed by /video-feed
latest_annotated = {"jpeg": None}
annotated_lock   = threading.Lock()

# Phone bounding boxes — written by phone_loop, overlaid by attention_loop
phone_boxes = {"boxes": [], "confs": []}
boxes_lock  = threading.Lock()

# Per-session statistics — reset on first start, read by /end-session
session_stats = {
    "start_time":         None,
    "phone_seconds":      0.0,
    "distracted_seconds": 0.0,
}
session_lock = threading.Lock()
