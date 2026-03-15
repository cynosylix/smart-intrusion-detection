"""
DeepFace-based face registration + recognition helpers.

- Face detection: OpenCV Haar Cascade
- Embeddings: DeepFace Facenet512
- Storage: embeddings/{person_name}.npy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace


@dataclass
class FaceConfig:
    embeddings_dir: str = "embeddings"
    model_name: str = "Facenet512"
    euclidean_threshold: float = 10.0  # tune if needed
    face_padding: int = 20
    min_face_size: Tuple[int, int] = (40, 40)


_MODEL = None
_CASCADE = None


def ensure_dirs(cfg: FaceConfig) -> None:
    os.makedirs(cfg.embeddings_dir, exist_ok=True)


def get_cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(cascade_path)
    return _CASCADE


def get_model(cfg: FaceConfig):
    """Build DeepFace model once for speed."""
    global _MODEL
    if _MODEL is None:
        _MODEL = DeepFace.build_model(cfg.model_name)
    return _MODEL


def detect_face_haar(rgb_img: np.ndarray, cfg: FaceConfig) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns a face box as (x1, y1, x2, y2) in the given rgb_img coordinates.
    Picks the largest detected face.
    """
    if rgb_img is None or rgb_img.size == 0:
        return None

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    cascade = get_cascade()
    if cascade.empty():
        return None

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=cfg.min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if faces is None or len(faces) == 0:
        # more lenient
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
    if faces is None or len(faces) == 0:
        return None

    # choose largest
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = cfg.face_padding
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(rgb_img.shape[1], x + w + pad)
    y2 = min(rgb_img.shape[0], y + h + pad)
    return (x1, y1, x2, y2)


def crop_rgb(rgb_img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return rgb_img[y1:y2, x1:x2]


def embedding_facenet512(rgb_face: np.ndarray, cfg: FaceConfig) -> Optional[np.ndarray]:
    """
    Returns a (512,) embedding vector.
    Uses enforce_detection=False because we already did face detection.
    """
    if rgb_face is None or rgb_face.size == 0:
        return None

    # Ensure model is built (warm-up)
    _ = get_model(cfg)

    rep = DeepFace.represent(
        img_path=rgb_face,
        model_name=cfg.model_name,
        enforce_detection=False,
    )
    # DeepFace may return list[dict] or dict depending on version
    if isinstance(rep, list):
        if not rep:
            return None
        emb = rep[0].get("embedding")
    else:
        emb = rep.get("embedding")

    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def save_embedding(person_name: str, emb: np.ndarray, cfg: FaceConfig) -> str:
    safe = "".join(c for c in person_name.strip() if c.isalnum() or c in ("_", "-")).strip()
    if not safe:
        raise ValueError("Invalid person name")
    ensure_dirs(cfg)
    path = os.path.join(cfg.embeddings_dir, f"{safe}.npy")
    np.save(path, emb)
    return path


def load_all_embeddings(cfg: FaceConfig) -> Dict[str, np.ndarray]:
    ensure_dirs(cfg)
    out: Dict[str, np.ndarray] = {}
    for fn in os.listdir(cfg.embeddings_dir):
        if fn.lower().endswith(".npy"):
            name = os.path.splitext(fn)[0]
            try:
                out[name] = np.load(os.path.join(cfg.embeddings_dir, fn)).astype(np.float32)
            except Exception:
                continue
    return out


def match_embedding(emb: np.ndarray, db: Dict[str, np.ndarray], cfg: FaceConfig) -> Tuple[Optional[str], float]:
    """
    Returns (best_name_or_None, best_distance). Lower distance is better.
    """
    best_name = None
    best_dist = float("inf")
    for name, ref in db.items():
        if ref is None:
            continue
        d = float(np.linalg.norm(ref - emb))
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_name is None:
        return None, best_dist
    if best_dist <= cfg.euclidean_threshold:
        return best_name, best_dist
    return None, best_dist

