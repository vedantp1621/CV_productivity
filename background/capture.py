import cv2
import queue as q

from background.shared import yolo_queue, attention_queue, monitoring_active


def _push(target_queue: q.Queue, frame):
    """Insert frame, dropping the oldest if the queue is full."""
    if target_queue.full():
        try:
            target_queue.get_nowait()
        except q.Empty:
            pass
    target_queue.put_nowait(frame)


def capture_loop(webcam_index: int = 0):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print("Capture: could not open webcam.")
        return

    # warm up — macOS needs a few dummy reads before frames arrive
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            break
    else:
        print("Capture: webcam failed to warm up.")
        cap.release()
        return

    try:
        while monitoring_active.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Capture: failed to grab frame.")
                break

            _push(yolo_queue, frame)
            _push(attention_queue, frame)
    finally:
        cap.release()
        monitoring_active.clear()
        print("Capture: webcam released.")
