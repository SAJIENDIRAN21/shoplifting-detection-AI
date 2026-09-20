"""Build normalized pose sequences from labelled videos, with a synthetic fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from utils.normalization import PoseSequenceBuffer, normalize_pose


PROJECT_DIR = Path(__file__).resolve().parent
RAW_VIDEO_DIR = PROJECT_DIR / "data" / "raw_videos"
SEQUENCE_DIR = PROJECT_DIR / "data" / "sequences"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def iter_videos(folder: Path) -> Iterable[Path]:
    return (path for path in sorted(folder.rglob("*")) if path.suffix.lower() in VIDEO_EXTENSIONS)


def extract_video_sequences(model: YOLO, video_path: Path, label: int) -> tuple[list[np.ndarray], list[int]]:
    """Sample one video at 15 FPS and collect per-track rolling sequences."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return [], []
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    sample_every = max(1, round(source_fps / 15.0))
    buffers = PoseSequenceBuffer()
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if (frame_number - 1) % sample_every:
                continue
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            boxes, keypoints = result.boxes, result.keypoints
            if boxes is None or keypoints is None or boxes.id is None or keypoints.data is None:
                continue
            ids = boxes.id.int().cpu().tolist()
            xywhs = boxes.xywh.cpu().numpy()
            poses = keypoints.data.cpu().numpy()
            for track_id, xywh, pose in zip(ids, xywhs, poses):
                sequence = buffers.append(track_id, normalize_pose(pose, xywh))
                # A stride of 15 prevents highly redundant near-duplicate windows.
                if sequence is not None and frame_number % (sample_every * 15) == 0:
                    sequences.append(sequence.copy())
                    labels.append(label)
    finally:
        capture.release()
    return sequences, labels


def generate_synthetic_sequences(samples: int = 5306, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, learnable pose motion data for a first runnable training pass.

    This is explicitly a development fallback, not a substitute for annotated
    surveillance footage. Class 1 includes a repeated hand-to-hip trajectory;
    class 0 uses low-amplitude neutral movement.
    """
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 1.0, 30, dtype=np.float32)
    data = np.empty((samples, 30, 51), dtype=np.float32)
    labels = np.arange(samples, dtype=np.int64) % 2
    rng.shuffle(labels)
    for index, label in enumerate(labels):
        pose = rng.normal(0.0, 0.035, size=(30, 17, 3)).astype(np.float32)
        pose[:, :, 2] = np.clip(rng.normal(0.88, 0.07, size=(30, 17)), 0.45, 1.0)
        # Use stable torso/hip landmarks as an anchor, like the real normalizer.
        pose[:, 11:13, :2] = rng.normal(0.0, 0.01, size=(30, 2, 2))
        if label:
            trajectory = 0.55 * np.sin(np.pi * time_axis) + rng.normal(0, 0.025, 30)
            for wrist in (9, 10):
                pose[:, wrist, 0] += trajectory * (1 if wrist == 10 else -1)
                pose[:, wrist, 1] += 0.20 * np.sin(2 * np.pi * time_axis)
            pose[:, 7:11, :2] += rng.normal(0.0, 0.07, size=(30, 4, 2))
        else:
            pose[:, :, :2] += 0.04 * np.sin(2 * np.pi * time_axis)[:, None, None]
        data[index] = pose.reshape(30, 51)
    return data, labels.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 30-frame YOLOv8-pose training sequences.")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt", help="Ultralytics pose model weights")
    parser.add_argument("--force-synthetic", action="store_true", help="Skip video extraction and generate development data")
    args = parser.parse_args()

    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    normal_dir, concealment_dir = RAW_VIDEO_DIR / "normal", RAW_VIDEO_DIR / "concealment"
    normal_dir.mkdir(parents=True, exist_ok=True)
    concealment_dir.mkdir(parents=True, exist_ok=True)
    all_sequences: list[np.ndarray] = []
    all_labels: list[int] = []
    videos = [(video, 0) for video in iter_videos(normal_dir)] + [(video, 1) for video in iter_videos(concealment_dir)]

    if videos and not args.force_synthetic:
        model = YOLO(args.pose_weights)
        for video, label in videos:
            print(f"Extracting {'concealment' if label else 'normal'}: {video.name}")
            sequences, labels = extract_video_sequences(model, video, label)
            all_sequences.extend(sequences)
            all_labels.extend(labels)

    if not all_sequences or len(set(all_labels)) < 2:
        print("No usable sequences for both classes found; generating 5,306 synthetic development sequences.")
        x_data, y_data = generate_synthetic_sequences()
    else:
        x_data = np.asarray(all_sequences, dtype=np.float32)
        y_data = np.asarray(all_labels, dtype=np.float32)
        print(f"Extracted {len(x_data)} sequences from {len(videos)} video(s).")

    np.save(SEQUENCE_DIR / "X.npy", x_data)
    np.save(SEQUENCE_DIR / "y.npy", y_data)
    print(f"Saved X{tuple(x_data.shape)} and y{tuple(y_data.shape)} to {SEQUENCE_DIR}")


if __name__ == "__main__":
    main()
