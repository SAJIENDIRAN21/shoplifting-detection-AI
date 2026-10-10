"""Pose feature normalization and per-person temporal buffering."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np


POSE_FEATURE_DIM = 51
SEQUENCE_LENGTH = 30


def normalize_pose(keypoints_xy: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray:
    """Normalize a COCO-17 pose relative to its hip center and bounding box.

    Parameters
    ----------
    keypoints_xy:
        Array with shape ``(17, 3)`` containing x, y, confidence values. A
        ``(17, 2)`` array is also accepted and receives confidence 1.0.
    bbox_xywh:
        Bounding box ``[center_x, center_y, width, height]`` (YOLO format).

    Returns
    -------
    numpy.ndarray
        Float32 vector in the order x_norm, y_norm, confidence for each of
        the 17 joints; its shape is always ``(51,)``.
    """
    points = np.asarray(keypoints_xy, dtype=np.float32)
    box = np.asarray(bbox_xywh, dtype=np.float32).reshape(-1)
    if points.shape not in ((17, 2), (17, 3)):
        raise ValueError("keypoints_xy must have shape (17, 2) or (17, 3)")
    if box.size != 4:
        raise ValueError("bbox_xywh must contain exactly four values")

    if points.shape[1] == 2:
        points = np.column_stack((points, np.ones(17, dtype=np.float32)))

    # COCO keypoint indices 11 and 12 correspond to left and right hip.
    hip_center = (points[11, :2] + points[12, :2]) / 2.0
    width = max(float(abs(box[2])), 1e-6)
    height = max(float(abs(box[3])), 1e-6)

    normalized = points.copy()
    normalized[:, 0] = (points[:, 0] - hip_center[0]) / width
    normalized[:, 1] = (points[:, 1] - hip_center[1]) / height
    normalized[:, 2] = np.clip(points[:, 2], 0.0, 1.0)
    return normalized.reshape(POSE_FEATURE_DIM).astype(np.float32, copy=False)


class PoseSequenceBuffer:
    """A FIFO buffer of up to 30 normalized poses for every active track."""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self._buffers: Dict[int, Deque[np.ndarray]] = {}

    def append(self, track_id: int, pose: np.ndarray) -> Optional[np.ndarray]:
        """Append one pose and return a complete sequence when available."""
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose.size != POSE_FEATURE_DIM:
            raise ValueError(f"pose must contain {POSE_FEATURE_DIM} features")
        track_id = int(track_id)
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.sequence_length)
        self._buffers[track_id].append(pose)
        if len(self._buffers[track_id]) < self.sequence_length:
            return None
        return np.stack(self._buffers[track_id]).astype(np.float32, copy=False)

    def clear(self, track_id: int) -> None:
        """Discard a departed person's stored frames."""
        self._buffers.pop(int(track_id), None)

    def clear_except(self, active_track_ids: set[int]) -> None:
        """Release buffers for tracks not seen in the current processing pass."""
        active = {int(track_id) for track_id in active_track_ids}
        for track_id in list(self._buffers):
            if track_id not in active:
                del self._buffers[track_id]
