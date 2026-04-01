import queue
import threading

# Separate queues per consumer so every frame reaches both pipelines
yolo_queue = queue.Queue(maxsize=10)
gaze_queue = queue.Queue(maxsize=10)

# Set to start monitoring, clear to stop all loops
monitoring_active = threading.Event()

# Latest detection results — written by consumer threads, read by /status
detection_state = {"phone": False, "eyes_off": False}
detection_lock = threading.Lock()
