import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

PHONE_THRESHOLD = 0.75
POSE_THRESHOLD = 0.65

_model = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model():
    global _model
    if _model is None:
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Drop classifier; keep features + avgpool → (1, 576, 1, 1) output
        _model = nn.Sequential(*list(base.children())[:-1])
        _model.eval()
    return _model


def embed_image(frame: np.ndarray) -> np.ndarray:
    """Return an L2-normalised 576-d feature vector for a BGR OpenCV frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)
    model = _load_model()
    with torch.no_grad():
        feat = model(tensor)        # (1, 576, 1, 1)
    vec = feat.squeeze().numpy()    # (576,)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two L2-normalised vectors."""
    return float(np.dot(a, b))
