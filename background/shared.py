import queue
import threading

# One queue per consumer so every frame reaches both pipelines
yolo_queue      = queue.Queue(maxsize=10)
attention_queue = queue.Queue(maxsize=10)

# Set to start monitoring, clear to stop all loops
monitoring_active = threading.Event()

# Latest detection results — written by consumer threads, read by /status
detection_state = {"phone": False, "eyes_off": False, "calibrating": False}
detection_lock  = threading.Lock()
