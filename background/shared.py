import queue
import threading

# Single queue for the unified detection pipeline
detection_queue = queue.Queue(maxsize=10)

# Set to start monitoring, clear to stop all loops
monitoring_active = threading.Event()

# Set after /configure succeeds
config_complete = threading.Event()

# Reference embeddings populated during the config phase
reference_embeddings = {"phone": [], "pose": []}

# Latest detection results — written by the detector thread, read by /status
detection_state = {"phone": False, "eyes_off": False}
detection_lock = threading.Lock()
