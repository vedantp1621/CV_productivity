import time
import cv2
import numpy as np
import mediapipe as mp
import queue as q

from background.shared import (
    attention_queue, monitoring_active,
    detection_state, detection_lock,
    session_stats, session_lock,
    latest_annotated, annotated_lock,
    phone_boxes, boxes_lock,
)

# ── Head pose thresholds (applied to deltas, not absolutes) ───────────────
YAW_THRESHOLD   = 25   # degrees from baseline
PITCH_THRESHOLD = 20   # degrees from baseline

# ── Temporal smoothing ────────────────────────────────────────────────────
SCORE_MAX       = 30
SCORE_THRESHOLD = 10
SCORE_INCREMENT = 1    # slow to trigger
SCORE_DECREMENT = 3    # fast to recover

# ── Calibration / adaptive baseline ──────────────────────────────────────
CALIB_FRAMES = 45
DEAD_ZONE    = 5.0
DRIFT_RATE   = 0.02

# ── MediaPipe landmark indices ────────────────────────────────────────────
#   4   = nose tip        152 = chin
#   33  = left eye outer  263 = right eye outer
#   61  = left mouth      291 = right mouth
_LANDMARK_IDS = [4, 152, 33, 263, 61, 291]

# ── Generic 3-D face model (mm, arbitrary scale) ─────────────────────────
_MODEL_POINTS = np.array([
    (   0.0,    0.0,    0.0),
    (   0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

_DIST_COEFFS = np.zeros((4, 1))

# ── Annotation colors (BGR) ───────────────────────────────────────────────
_C_GREEN = ( 80, 210,  80)
_C_RED   = ( 60,  60, 220)
_C_AMBER = ( 20, 160, 245)
_C_CYAN  = (210, 220,  50)
_C_WHITE = (200, 200, 200)
_C_BLACK = (  0,   0,   0)
_AXIS_X  = ( 60,  60, 220)   # red  (BGR)
_AXIS_Y  = ( 60, 200,  60)   # green
_AXIS_Z  = (200, 120,  40)   # blue-ish


def _solve_pose(landmarks, w: int, h: int):
    """Return (pitch, yaw, rvec, tvec, cam_matrix) or all-None on failure."""
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
    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, img_pts, cam, _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) - 180.0   # normalise: straight-ahead → 0°
    yaw   = float(angles[1])
    return pitch, yaw, rvec, tvec, cam


def _annotate(frame, lm, rvec, tvec, cam,
              delta_pitch, delta_yaw, score,
              calibrated, calibrating, calib_count, final_distracted):
    """Draw CV overlay and return annotated copy. lm/rvec/tvec may be None."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Status border ─────────────────────────────────────────────────────
    if calibrating:
        border = _C_AMBER
    elif final_distracted:
        border = _C_RED
    else:
        border = _C_GREEN
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border, 6)

    if lm is not None:
        # ── Landmark dots ─────────────────────────────────────────────────
        for idx in _LANDMARK_IDS:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            cv2.circle(vis, (px, py), 5, _C_CYAN, -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 5, _C_BLACK, 1,  cv2.LINE_AA)

        # ── 3-D pose axes from nose tip ───────────────────────────────────
        if calibrated and rvec is not None:
            nose = (int(lm[4].x * w), int(lm[4].y * h))
            axis_3d = np.float32([[80, 0, 0], [0, -80, 0], [0, 0, 80]])
            proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam, _DIST_COEFFS)
            for pt, color in zip(proj, [_AXIS_X, _AXIS_Y, _AXIS_Z]):
                end = tuple(map(int, pt.ravel()))
                cv2.arrowedLine(vis, nose, end, color, 2,
                                tipLength=0.25, line_type=cv2.LINE_AA)

        # ── Text overlay ──────────────────────────────────────────────────
        if calibrated:
            focus_color = _C_RED if final_distracted else _C_GREEN
            txt = f"dP {delta_pitch:+.1f}   dY {delta_yaw:+.1f}"
            cv2.putText(vis, txt, (10, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, focus_color, 1, cv2.LINE_AA)
        elif calibrating:
            pct = int((calib_count / CALIB_FRAMES) * 100)
            cv2.putText(vis, f"CALIBRATING  {pct}%", (10, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _C_AMBER, 1, cv2.LINE_AA)

    # ── Temporal score bar ────────────────────────────────────────────────
    bar_max_w = w - 20
    bar_filled = int((score / SCORE_MAX) * bar_max_w)
    thresh_x   = 10 + int((SCORE_THRESHOLD / SCORE_MAX) * bar_max_w)
    bar_color  = _C_RED if final_distracted else _C_GREEN
    cv2.rectangle(vis, (10, h - 11), (w - 10, h - 5), (40, 40, 40), -1)
    if bar_filled > 0:
        cv2.rectangle(vis, (10, h - 11), (10 + bar_filled, h - 5),
                      bar_color, -1, cv2.LINE_AA)
    # threshold marker
    cv2.line(vis, (thresh_x, h - 13), (thresh_x, h - 3), _C_WHITE, 1, cv2.LINE_AA)

    # ── Phone bounding boxes ──────────────────────────────────────────────
    with boxes_lock:
        bxs = list(phone_boxes["boxes"])
        cfs = list(phone_boxes["confs"])

    for box, conf in zip(bxs, cfs):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), _C_AMBER, 2, cv2.LINE_AA)
        label = f"Phone  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 8, y1), _C_AMBER, -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_BLACK, 1, cv2.LINE_AA)

    return vis


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
    delta_pitch   = 0.0
    delta_yaw     = 0.0

    calib_yaws    = []
    calib_pitches = []
    calibrated    = False
    baseline_yaw  = 0.0
    baseline_pitch = 0.0

    last_t         = None
    was_distracted = False

    print("Attention: thread started.")

    while monitoring_active.is_set():
        try:
            frame = attention_queue.get(timeout=1)
        except q.Empty:
            last_t = None
            continue

        # ── Time accounting ───────────────────────────────────────────────
        now = time.monotonic()
        if last_t is not None:
            dt = now - last_t
            if was_distracted:
                with session_lock:
                    session_stats["distracted_seconds"] += dt
        last_t = now

        # ── Head pose detection ────────────────────────────────────────────
        h_px, w_px = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lm = rvec = tvec = cam_mat = None

        if not result.multi_face_landmarks:
            distracted = has_seen_face
        else:
            if not has_seen_face:
                with detection_lock:
                    detection_state["calibrating"] = True
            has_seen_face = True
            lm = result.multi_face_landmarks[0].landmark

            pitch, yaw, rvec, tvec, cam_mat = _solve_pose(lm, w_px, h_px)

            if pitch is None:
                distracted = False
            elif not calibrated:
                calib_yaws.append(yaw)
                calib_pitches.append(pitch)
                distracted = False
                if len(calib_yaws) >= CALIB_FRAMES:
                    baseline_yaw   = float(np.mean(calib_yaws))
                    baseline_pitch = float(np.mean(calib_pitches))
                    calibrated = True
                    with detection_lock:
                        detection_state["calibrating"] = False
                    print(f"[calib] baseline — pitch={baseline_pitch:.1f}° yaw={baseline_yaw:.1f}°")
            else:
                delta_yaw   = yaw   - baseline_yaw
                delta_pitch = pitch - baseline_pitch

                if abs(delta_yaw)   < DEAD_ZONE: delta_yaw   = 0.0
                if abs(delta_pitch) < DEAD_ZONE: delta_pitch = 0.0

                distracted = (abs(delta_yaw) > YAW_THRESHOLD or
                              abs(delta_pitch) > PITCH_THRESHOLD)

                if not distracted:
                    baseline_yaw   = (1 - DRIFT_RATE) * baseline_yaw   + DRIFT_RATE * yaw
                    baseline_pitch = (1 - DRIFT_RATE) * baseline_pitch + DRIFT_RATE * pitch

                _tick += 1
                if _tick % 30 == 0:
                    print(f"[pose] Δpitch={delta_pitch:.1f}° Δyaw={delta_yaw:.1f}°  "
                          f"distracted={distracted}")

        # ── Temporal smoothing ─────────────────────────────────────────────
        score = score + SCORE_INCREMENT if distracted else score - SCORE_DECREMENT
        score = max(0, min(score, SCORE_MAX))

        final_distracted = score > SCORE_THRESHOLD
        was_distracted   = final_distracted
        calibrating      = has_seen_face and not calibrated

        with detection_lock:
            detection_state["eyes_off"]    = final_distracted
            detection_state["delta_pitch"] = round(delta_pitch, 1)
            detection_state["delta_yaw"]   = round(delta_yaw, 1)
            detection_state["score"]       = score

        # ── Annotate frame and publish ─────────────────────────────────────
        vis = _annotate(frame, lm, rvec, tvec, cam_mat,
                        delta_pitch, delta_yaw, score,
                        calibrated, calibrating, len(calib_yaws),
                        final_distracted)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with annotated_lock:
                latest_annotated["jpeg"] = buf.tobytes()

    # Cleanup
    with detection_lock:
        detection_state["eyes_off"]    = False
        detection_state["calibrating"] = False
        detection_state["delta_pitch"] = 0.0
        detection_state["delta_yaw"]   = 0.0
        detection_state["score"]       = 0
    with annotated_lock:
        latest_annotated["jpeg"] = None

    face_mesh.close()
    print("Attention: thread stopped.")
