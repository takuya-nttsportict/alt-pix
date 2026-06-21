"""Team classification: cluster on-court players into two teams.

Approach (roboflow/sports lineage, all commercial-friendly):
  1. Crop each tracked player's torso region.
  2. Embed the crop:
       - "siglip"  : google/siglip-base-patch16-224 image embedding (Apache 2.0).
                     Robust to lighting / viewpoint, no per-video training.
       - "hsv"     : hue/saturation histogram of the torso (numpy only).
                     Lightweight fallback when transformers/torch are absent.
  3. KMeans(k=2) over a warm-up buffer of embeddings → two team centroids.
  4. Each subsequent frame assigns players to the nearest centroid.

Why this generalises (principle 4/5 in CLAUDE.md):
  - Embedding + clustering is *relative* to the uniforms actually present, so it
    does not over-fit a particular venue, camera angle, or jersey colour. A new
    match self-calibrates during warm-up.
  - No dataset with commercial-use restrictions is required (SigLIP weights are
    Apache 2.0; the HSV path needs no weights at all).

The classifier is deliberately split into:
  - a stateless `fit` / `predict` over embeddings (unit-testable, numpy-only),
  - a stateful `update(frame, tracks)` wrapper that buffers crops during warm-up
    and then labels every frame, mirroring `JerseyOCR.update`.

`team` is an opaque integer (0 or 1); which physical team it maps to is stable
within a session but arbitrary (KMeans label order is not meaningful).
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np

from .tracker import Track

logger = logging.getLogger(__name__)

# Torso region as a fraction of the player bbox. Volleyball jerseys are on the
# upper body; legs/shorts and the floor add noise, so we sample the chest band.
_TORSO_TOP = 0.15
_TORSO_BOT = 0.50
_TORSO_SIDE = 0.15  # trim left/right margins (arms, background)

_HSV_H_BINS = 12
_HSV_S_BINS = 4


def _torso_crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Extract the torso sub-image from a player bbox. Returns None if degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    cx1 = int(round(x1 + _TORSO_SIDE * bw))
    cx2 = int(round(x2 - _TORSO_SIDE * bw))
    cy1 = int(round(y1 + _TORSO_TOP * bh))
    cy2 = int(round(y1 + _TORSO_BOT * bh))
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return frame[cy1:cy2, cx1:cx2]


def _hsv_embed(crop: np.ndarray) -> np.ndarray:
    """Normalised hue-saturation histogram of a BGR torso crop (numpy fallback)."""
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_HSV_H_BINS, _HSV_S_BINS],
                        [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else hist


# ── Stateless KMeans (numpy only) ───────────────────────────────────────────────

def _kmeans2(X: np.ndarray, n_iters: int = 25, seed: int = 0) -> np.ndarray:
    """2-means via k-means++ init. Returns centroids (2, D).

    Kept dependency-free and deterministic so team assignment is reproducible
    and unit-testable without sklearn (principle 6: designed dependencies).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 2:
        # Degenerate: duplicate the only point so predict() still works.
        c = X[0] if n == 1 else np.zeros(X.shape[1])
        return np.stack([c, c])

    # k-means++ seeding for the 2nd centroid.
    i0 = int(rng.integers(n))
    c0 = X[i0]
    d2 = np.sum((X - c0) ** 2, axis=1)
    probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
    i1 = int(rng.choice(n, p=probs))
    centroids = np.stack([c0, X[i1]]).astype(np.float64)

    for _ in range(n_iters):
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n, 2)
        labels = d.argmin(axis=1)
        new = centroids.copy()
        for k in (0, 1):
            mask = labels == k
            if mask.any():
                new[k] = X[mask].mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids


class TeamClassifier:
    """Two-team uniform clustering with a warm-up buffer.

    Args:
        backend:    "siglip" (default, needs transformers+torch) or "hsv" (numpy
                    fallback). On siglip import/load failure it falls back to hsv.
        warmup_frames: collect crops for this many *processed* frames before the
                    first KMeans fit. The buffer keeps accumulating and the model
                    is refit periodically so late-arriving uniforms are captured.
        refit_every: refit KMeans every N frames after warm-up (0 = never refit).
        device:     torch device for the SigLIP backend.
    """

    def __init__(
        self,
        backend: Literal["siglip", "hsv"] = "siglip",
        warmup_frames: int = 30,
        refit_every: int = 150,
        device: str = "cuda",
    ) -> None:
        self._backend = backend
        self._warmup = warmup_frames
        self._refit_every = refit_every
        self._device = device

        self._centroids: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []
        self._frames_seen = 0
        self._siglip = None  # lazy

        if backend == "siglip":
            self._try_load_siglip()

    # ── Embedding backends ──────────────────────────────────────────────────────

    def _try_load_siglip(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            name = "google/siglip-base-patch16-224"
            self._proc = AutoProcessor.from_pretrained(name)
            self._siglip = AutoModel.from_pretrained(name).to(self._device).eval()
            self._torch = torch
            logger.info(f"TeamClassifier: SigLIP backend loaded ({name})")
        except Exception as e:  # noqa: BLE001 — degrade, don't crash the pipeline
            logger.warning(
                f"TeamClassifier: SigLIP unavailable ({e!r}); falling back to HSV histogram."
            )
            self._backend = "hsv"
            self._siglip = None

    def _embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0))
        if self._backend == "siglip" and self._siglip is not None:
            return self._embed_siglip(crops)
        return np.stack([_hsv_embed(c) for c in crops])

    def _embed_siglip(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        inputs = self._proc(images=rgb, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            feats = self._siglip.get_image_features(**inputs)
        feats = feats.cpu().numpy().astype(np.float64)
        # L2-normalise so KMeans clusters by direction (colour/pattern), not scale.
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norm, 1e-8, None)

    # ── Stateless fit / predict (unit-testable) ─────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the two team centroids from a matrix of embeddings (N, D)."""
        if len(embeddings) < 2:
            logger.debug("TeamClassifier.fit: <2 samples, deferring.")
            return
        self._centroids = _kmeans2(np.asarray(embeddings, dtype=np.float64))
        logger.info(f"TeamClassifier: fit 2 team centroids on {len(embeddings)} samples.")

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign each embedding to team 0/1. Returns int array (N,)."""
        if self._centroids is None or len(embeddings) == 0:
            return np.full(len(embeddings), -1, dtype=int)
        d = np.linalg.norm(
            np.asarray(embeddings)[:, None, :] - self._centroids[None, :, :], axis=2
        )
        return d.argmin(axis=1)

    def distance_to_teams(self, embeddings: np.ndarray) -> np.ndarray:
        """Min distance of each embedding to its nearest team centroid (N,).

        Used downstream for referee/outlier detection (a non-team uniform sits
        far from both centroids).
        """
        if self._centroids is None or len(embeddings) == 0:
            return np.full(len(embeddings), np.inf)
        d = np.linalg.norm(
            np.asarray(embeddings)[:, None, :] - self._centroids[None, :, :], axis=2
        )
        return d.min(axis=1)

    @property
    def ready(self) -> bool:
        return self._centroids is not None

    # ── Stateful per-frame wrapper ──────────────────────────────────────────────

    def update(
        self, frame: np.ndarray, tracks: list[Track]
    ) -> tuple[dict[int, int], dict[int, float]]:
        """Buffer crops during warm-up, then label every track.

        Returns:
            team_map: {track_id: team(0/1) or -1 if not yet ready}
            dist_map: {track_id: distance to nearest team centroid} (for role/outlier)
        """
        self._frames_seen += 1

        crops: list[np.ndarray] = []
        ids: list[int] = []
        for t in tracks:
            crop = _torso_crop(frame, t.bbox)
            if crop is not None:
                crops.append(crop)
                ids.append(t.track_id)

        if not crops:
            return {}, {}

        emb = self._embed(crops)

        # Warm-up: accumulate, fit once enough frames/samples are collected.
        if not self.ready:
            self._buffer.extend(emb)
            if self._frames_seen >= self._warmup and len(self._buffer) >= 4:
                self.fit(np.stack(self._buffer))
                self._buffer.clear()
            return {tid: -1 for tid in ids}, {}

        # Periodic refit to absorb uniforms unseen during warm-up.
        if self._refit_every and self._frames_seen % self._refit_every == 0:
            self._buffer.extend(emb)
            self.fit(np.stack(self._buffer[-512:]))  # cap buffer memory
            self._buffer = self._buffer[-512:]

        labels = self.predict(emb)
        dists = self.distance_to_teams(emb)
        team_map = {tid: int(lbl) for tid, lbl in zip(ids, labels)}
        dist_map = {tid: float(dd) for tid, dd in zip(ids, dists)}
        return team_map, dist_map
