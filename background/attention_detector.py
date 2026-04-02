import cv2
import numpy as np
import mediapipe as mp
import queue as q

from background.shared import attention_queue, monitoring_active, detection_state, detection_lock

# ── Head pose thresholds (applied to deltas, not absolutes) ───────────────
YAW_THRESHOLD   = 25   # degrees from baseline
PITCH_THRESHOLD = 20   # degrees from baseline

# ── Temporal smoothing ────────────────────────────────────────────────────
SCORE_MAX        = 30
SCORE_THRESHOLD  = 10
SCORE_INCREMENT  = 1   # slow to trigger
SCORE_DECREMENT  = 3   # fast to recover

# ── Calibration / adaptive baseline ──────────────────────────────────────
CALIB_FRAMES = 45     # frames to average for initial baseline (~1.5 s at 30 fps)
DEAD_ZONE    = 5.0    # degrees: deltas smaller than this are treated as zero
DRIFT_RATE   = 0.02   # EMA weight for baseline drift during focused periods

# ── MediaPipe landmark indices ────────────────────────────────────────────
#   4   = nose tip
#   152 = chin
#   33  = left eye outer corner  (camera-left)
#   263 = right eye outer corner (camera-right)
#   61  = left mouth corner
#   291 = right mouth corner
_LANDMARK_IDS = [4, 152, 33, 263, 61, 291]

# ── Generic 3-D face model (mm, arbitrary scale) ─────────────────────────
_MODEL_POINTS = np.array([
    (   0.0,    0.0,    0.0),   # nose tip
    (   0.0, -330.0,  -65.0),   # chin
    (-225.0,  170.0, -135.0),   # left eye outer corner
    ( 225.0,  170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0),   # left mouth corner
    ( 150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)

_DIST_COEFFS = np.zeros((4, 1))


def _head_angles(landmarks, w: int, h: int):
    """Return (pitch_deg, yaw_deg) using solvePnP, or (None, None) on failure."""
    img_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in _LANDMARK_IDS],
        dtype=np.float64,
    )

    focal = float(w)
    cam = np.array(
        [[focal, 0.0, w / 2.0],
         [0.0, focal, h / 2.0],
         [0.0,   0.0,     1.0]],
        dtype=np.float64,
    )

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, img_pts, cam, _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    # RQDecomp3x3 returns pitch ~180° when facing the camera (model Y-axis
    # is inverted vs OpenCV camera frame). Subtract 180 so straight-ahead → 0°.
    pitch = float(angles[0]) - 180.0
    yaw   = float(angles[1])
    return pitch, yaw


def attention_loop():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    score         = 0
    has_seen_face = False
    _tick         = 0

    # Calibration state
    calib_yaws    = []
    calib_pitches = []
    calibrated    = False
    baseline_yaw  = 0.0
    baseline_pitch= 0.0

    print("Attention: thread started.")

    while monitoring_active.is_set():
        try:
            frame = attention_queue.get(timeout=1)
        except q.Empty:
            continue

        h, w = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not result.multi_face_landmarks:
            distracted = has_seen_face  # no face = distracted only after first detection
        else:
            if not has_seen_face:
                # First frame with a confirmed face — start calibration
                with detection_lock:
                    detection_state["calibrating"] = True
            has_seen_face = True

            lm = result.multi_face_landmarks[0].landmark
            pitch, yaw = _head_angles(lm, w, h)

            if pitch is None:
                distracted = False  # solver failed — don't penalise

            elif not calibrated:
                # ── Calibration phase: collect baseline samples ───────────
                calib_yaws.append(yaw)
                calib_pitches.append(pitch)
                distracted = False  # never flag during calibration

                if len(calib_yaws) >= CALIB_FRAMES:
                    baseline_yaw   = float(np.mean(calib_yaws))
                    baseline_pitch = float(np.mean(calib_pitches))
                    calibrated = True
                    with detection_lock:
                        detection_state["calibrating"] = False
                    print(f"[calib] baseline — pitch={baseline_pitch:.1f}° yaw={baseline_yaw:.1f}°")

            else:
                # ── Active detection: compare delta vs baseline ───────────
                delta_yaw   = yaw   - baseline_yaw
                delta_pitch = pitch - baseline_pitch

                # Dead zone: absorb noise and micro-movements
                if abs(delta_yaw)   < DEAD_ZONE: delta_yaw   = 0.0
                if abs(delta_pitch) < DEAD_ZONE: delta_pitch = 0.0

                distracted = abs(delta_yaw) > YAW_THRESHOLD or abs(delta_pitch) > PITCH_THRESHOLD

                # Adaptive drift: baseline slowly follows focused posture
                if not distracted:
                    baseline_yaw   = (1.0 - DRIFT_RATE) * baseline_yaw   + DRIFT_RATE * yaw
                    baseline_pitch = (1.0 - DRIFT_RATE) * baseline_pitch + DRIFT_RATE * pitch

                _tick += 1
                if _tick % 30 == 0:
                    print(f"[pose] Δpitch={delta_pitch:.1f}° Δyaw={delta_yaw:.1f}°  "
                          f"distracted={distracted}")

        # Temporal smoothing
        score = score + SCORE_INCREMENT if distracted else score - SCORE_DECREMENT
        score = max(0, min(score, SCORE_MAX))

        with detection_lock:
            detection_state["eyes_off"] = score > SCORE_THRESHOLD

    with detection_lock:
        detection_state["eyes_off"]   = False
        detection_state["calibrating"] = False

    face_mesh.close()
    print("Attention: thread stopped.")
