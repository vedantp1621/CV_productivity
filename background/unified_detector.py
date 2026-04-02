import queue as q

from background.shared import (
    detection_queue,
    monitoring_active,
    detection_state,
    detection_lock,
    reference_embeddings,
)
from model.embedder import embed_image, cosine_similarity, PHONE_THRESHOLD, POSE_THRESHOLD


_tick = 0

def unified_loop():
    global _tick
    print("Detector: thread started.")
    while monitoring_active.is_set():
        try:
            frame = detection_queue.get(timeout=1)
        except q.Empty:
            continue

        emb = embed_image(frame)

        phone_refs = reference_embeddings["phone"]
        pose_refs = reference_embeddings["pose"]

        phone_sims    = [cosine_similarity(emb, r) for r in phone_refs]
        pose_sims     = [cosine_similarity(emb, r) for r in pose_refs]
        max_phone_sim = max(phone_sims, default=0.0)
        max_pose_sim  = max(pose_sims,  default=0.0)

        _tick += 1
        if _tick % 30 == 0:
            print(f"[sim] phone={max_phone_sim:.3f} (thresh={PHONE_THRESHOLD})  "
                  f"pose={max_pose_sim:.3f} (thresh={POSE_THRESHOLD})")

        phone     = max_phone_sim > PHONE_THRESHOLD
        distracted = max_pose_sim  > POSE_THRESHOLD

        with detection_lock:
            detection_state["phone"] = phone
            detection_state["eyes_off"] = distracted

    with detection_lock:
        detection_state["phone"] = False
        detection_state["eyes_off"] = False
    print("Detector: thread stopped.")
