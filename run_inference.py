"""
Real-time CCTV Shoplifting Detection & Behavioral Analytics Engine
==================================================================
Standalone, pose-only. Dependencies: ultralytics (brings torch), opencv-python, numpy.

Two-Tier Intelligence System
  Tier 1  WARNING  : hands linger in the waist / pocket / torso zone for > 25 frames
                     BROWSING -> SUSPICIOUS (orange) + warning log
  Tier 2  THEFT    : the same behaviour persists / escalates to > 60 frames
                     SUSPICIOUS -> THEFT (red) + alarm print + forensic export to evidence/
                     (full-scene snapshot with the suspect boxed + upscaled suspect crop)

Why it does not depend on the item detector
  Items are never detected, so unknown grocery / snack items cannot be missed. The decision rests on
  skeletal kinematics: hand-to-torso proximity, a bent-arm / centre-line gesture test (arms hanging
  straight at the sides do not count), a recent reach toward the shelf, and dwell time.

Top-down / ceiling CCTV robustness
  The waist is never "unavailable". It is estimated from, in order of reliability:
    KPT  (hips + shoulders)  ->  HIP (hips only)  ->  SHOULDER (shoulders only)  ->  BOX (bbox ratios)
  and smoothed in box-relative coordinates, so switching source never makes the zone jump or the
  dwell timer break. Wrist keypoints that drop out are remembered for a few frames, and the dwell
  timer decays instead of resetting to zero.

Customer identity
  ByteTrack (or BoT-SORT) raw track IDs are mapped onto permanent "Customer N" IDs. A raw ID must
  survive a few frames to become a customer, and a lost customer is re-linked to a new track using
  motion-predicted position, size and a clothing-colour histogram.
"""

import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# Tunable constants
# ----------------------------------------------------------------------------------------------
# --- tiers (frames) ---
WARN_FRAMES = 45          # Increase from 25 -> 45 (requires ~1.5 seconds of sustained posture)
THEFT_FRAMES = 180        # Increase from 60 -> 180 (requires a full 6 seconds of continuous gesture)
RELEASE_DWELL = 10        
DWELL_GAP_TOL = 2         # Reduce from 3 -> 2 (shorter tolerance for brief keypoint dropouts)
DWELL_DECAY = 4           # Increase from 2 -> 4 (dwell drains faster when hands leave the zone)

# --- keypoint confidences ---
HIP_CONF = 0.45           # Raise from 0.30 -> 0.45 (only trust hips when clearly visible)
SHOULDER_CONF = 0.45      # Raise from 0.30 -> 0.45
WRIST_CONF = 0.35         # Raise from 0.25 -> 0.35 (ignore guessed or occluded hand locations)
ELBOW_CONF = 0.35         # Raise from 0.25 -> 0.35
WRIST_GRACE_FRAMES = 3    # Reduce from 6 -> 3 (stop remembering lost wrists too quickly)

# --- waist / torso zone, in torso lengths relative to the (smoothed) waist centre ---
ZONE_HALF_W = 0.55        # Narrower width to prevent side false alarms
ZONE_TOP = -0.40          # Tighter upper boundary
ZONE_BOTTOM = 0.45        # Tighter lower boundary
CENTER_DX = 0.25          # Stricter center-line check
ARM_STRAIGHT_DEG = 140.0  # <--- Globally declared here to fix NameError permanently
UNKNOWN_ELBOW_DX = 0.45
WAIST_EMA = 0.40

# --- shelf-reach gate (a concealment is preceded by reaching toward the shelf) ---
REACH_DIST = 1.40         # wrist farther than this (torso lengths) from the waist = arm extended
REACH_UP = -1.10          # ... or above the shoulder line
ENGAGE_WINDOW = 200       # frames after a reach during which waist dwell is suspicious (~10 s)

# --- customer identity layer ---
TRACK_BUFFER = 90
CONFIRM_FRAMES = 3        # a raw track must live this long before it becomes a customer
PENDING_TTL = 5
DUP_IOU = 0.70            # an unconfirmed track on top of a known customer is a duplicate detection
REID_MAX_LOST_FRAMES = 240
REID_APP_MAX = 0.55       # max clothing-histogram distance for a re-link (0 = identical)
REID_BASE_DIST = 0.60     # allowed jump in person-heights right after loss ...
REID_DIST_GROWTH = 0.04   # ... growing per lost frame ...
REID_DIST_CAP = 3.5       # ... up to this cap
REID_NO_APP_DIST = 0.80
REID_MAX_SIZE_RATIO = 2.5
EXTRAP_MAX_FRAMES = 30
RETAIN_LOST_FRAMES = 300  # forget non-suspect customers unseen this long
APP_UPDATE_ALPHA = 0.10
APP_OVERLAP_IOU = 0.15

NEVER = -10 ** 9

# BGR colours
COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 255)

STATE_COLOR = {"BROWSING": COLOR_GREEN, "SUSPICIOUS": COLOR_ORANGE, "THEFT": COLOR_RED}

# COCO-17 keypoints: (wrist, elbow, shoulder) per arm
ARMS = ((9, 7, 5), (10, 8, 6))
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]

# Tracker configs, written to trackers/<name>_retail.yaml (long buffer, permissive thresholds for 360p CCTV)
TRACKER_CFGS = {
    "bytetrack": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": TRACK_BUFFER,
        "match_thresh": 0.80,
        "fuse_score": True,
    },
    "botsort": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": TRACK_BUFFER,
        "match_thresh": 0.80,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.50,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "model": "auto",
    },
}


@dataclass
class Params:
    warn_frames: int = WARN_FRAMES
    theft_frames: int = THEFT_FRAMES
    reach_gate: bool = True


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------
def box_height(box: np.ndarray) -> float:
    return max(float(box[3] - box[1]), 1.0)


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b (degrees) between segments b->a and b->c."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / n, -1.0, 1.0))))


def estimate_waist(kpts: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Waist centre (pixels), torso length (pixels) and the source used. NEVER returns None, so the
    zone and the dwell timer survive hips hidden by a ceiling-mounted camera angle.

      KPT      hips + shoulders       exact torso length
      HIP      hips only              torso length from the box (0.30 * height)
      SHOULDER shoulders only         waist = shoulder-mid + one box-estimated torso length down
      BOX      nothing usable         waist at 55 % of the box height, centred
    """
    h = box_height(box)
    cx = (float(box[0]) + float(box[2])) / 2.0

    hips = [kpts[i][:2] for i in (L_HIP, R_HIP) if kpts[i][2] > HIP_CONF]
    shoulders = [kpts[i][:2] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] > SHOULDER_CONF]
    hip_mid = np.mean(hips, axis=0) if hips else None
    sh_mid = np.mean(shoulders, axis=0) if shoulders else None

    # Hips above the shoulders = hallucinated pose: ignore the hips
    if hip_mid is not None and sh_mid is not None and hip_mid[1] <= sh_mid[1]:
        hip_mid = None

    if hip_mid is not None and sh_mid is not None:
        center, torso, src = hip_mid, float(np.linalg.norm(sh_mid - hip_mid)), "KPT"
    elif hip_mid is not None:
        center, torso, src = hip_mid, 0.30 * h, "HIP"
    elif sh_mid is not None:
        torso = 0.30 * h
        center, src = np.array([sh_mid[0], sh_mid[1] + torso]), "SHOULDER"
    else:
        torso = 0.30 * h
        center, src = np.array([cx, float(box[1]) + 0.55 * h]), "BOX"

    torso = float(min(max(torso, 0.15 * h), 0.45 * h))
    return np.asarray(center, dtype=np.float32), torso, src


# ----------------------------------------------------------------------------------------------
# Appearance signature for identity re-linking
# ----------------------------------------------------------------------------------------------
def appearance_hist(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """L1-normalised 16x8 Hue/Saturation histogram of the upper body (V ignored: lighting-robust)."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(fw, int(box[2])), min(fh, int(box[3]))
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    crop = frame[y1 + int(0.15 * h): y1 + int(0.55 * h), x1 + int(0.15 * w): x2 - int(0.15 * w)]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    total = float(hist.sum())
    return hist / total if total > 1e-6 else None


def hist_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya distance in [0, 1]; None when either signature is missing."""
    if a is None or b is None:
        return None
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


# ----------------------------------------------------------------------------------------------
# Per-customer state
# ----------------------------------------------------------------------------------------------
@dataclass
class Customer:
    cid: int
    raw_id: Optional[int] = None
    present: bool = False
    first_seen: int = 0
    last_seen: int = 0
    box: Optional[np.ndarray] = None
    center_hist: deque = field(default_factory=lambda: deque(maxlen=8))   # (frame, cx, cy)
    app_hist: Optional[np.ndarray] = None

    # skeletal analysis
    waist_rel: Optional[Tuple[float, float, float]] = None   # (off_x/h, off_y/h, torso/h) smoothed
    waist_src: str = "BOX"
    wrist_mem: List[Optional[tuple]] = field(default_factory=lambda: [None, None])   # (dx, dy, ok, age)
    hands: List[np.ndarray] = field(default_factory=list)
    zone: Optional[Tuple[float, float, float, float]] = None  # pixel rectangle of the waist zone
    in_zone: bool = False
    last_reach: int = NEVER

    # dwell + tiers
    dwell: int = 0
    gap: int = 0
    state: str = "BROWSING"
    warned: bool = False
    locked: bool = False
    theft_frame: int = NEVER

    def to_dict(self, fps: float, theft_frames: int = THEFT_FRAMES) -> dict:
        """Plain dictionary view: {"dwell_time": s, "anomaly_score": f, "state": str, ...}"""
        return {
            "dwell_time": round(self.dwell / max(fps, 1.0), 2),
            "dwell_frames": self.dwell,
            "anomaly_score": round(min(1.0, self.dwell / float(max(theft_frames, 1))), 3),
            "state": self.state,
            "waist_source": self.waist_src,
            "present": self.present,
            "raw_track_id": self.raw_id,
        }


def predicted_box(c: Customer, frame_idx: int) -> np.ndarray:
    """Last box shifted by the customer's recent velocity (bounded extrapolation)."""
    box = c.box.astype(np.float32).copy()
    if len(c.center_hist) >= 2:
        f0, x0, y0 = c.center_hist[0]
        f1, x1, y1 = c.center_hist[-1]
        if f1 > f0:
            vx, vy = (x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0)
            age = min(max(frame_idx - c.last_seen, 0), EXTRAP_MAX_FRAMES)
            box += np.array([vx * age, vy * age, vx * age, vy * age], dtype=np.float32)
    return box


class CustomerRegistry:
    """Maps volatile tracker IDs to permanent customer IDs and owns every customer's state."""

    def __init__(self, fps: float, theft_frames: int = THEFT_FRAMES):
        self.fps = fps if fps and fps > 1 else 20.0
        self.theft_frames = theft_frames
        self.customers: Dict[int, Customer] = {}
        self.raw_to_cid: Dict[int, int] = {}
        self.cid_to_raw: Dict[int, int] = {}
        self.pending: Dict[int, dict] = {}
        self.next_id = 1

    @property
    def active_customers(self) -> Dict[int, Customer]:
        return {cid: c for cid, c in self.customers.items() if c.present}

    def snapshot(self, only_active: bool = True) -> Dict[int, dict]:
        """active_customers-style dictionary: { id: {"dwell_time", "anomaly_score", "state", ...} }"""
        return {cid: c.to_dict(self.fps, self.theft_frames) for cid, c in self.customers.items()
                if c.present or not only_active}

    def _link(self, cid: int, raw: int) -> None:
        old_raw = self.cid_to_raw.get(cid)
        if old_raw is not None:
            self.raw_to_cid.pop(old_raw, None)
        old_cid = self.raw_to_cid.get(raw)
        if old_cid is not None and old_cid != cid:
            self.cid_to_raw.pop(old_cid, None)
        self.raw_to_cid[raw] = cid
        self.cid_to_raw[cid] = raw
        self.customers[cid].raw_id = raw

    def _relink_cost(self, frame_idx: int, c: Customer, box: np.ndarray,
                     hist: Optional[np.ndarray], age: int) -> Optional[float]:
        pred = predicted_box(c, frame_idx)
        h_old, h_new = box_height(c.box), box_height(box)
        if max(h_old, h_new) / min(h_old, h_new) > REID_MAX_SIZE_RATIO:
            return None
        dist = float(np.linalg.norm(box_center(pred) - box_center(box))) / max(h_old, h_new)
        max_d = min(REID_DIST_CAP, REID_BASE_DIST + REID_DIST_GROWTH * age)
        if dist > max_d:
            return None
        app = hist_distance(c.app_hist, hist)
        if app is not None and app > REID_APP_MAX:
            return None
        if app is None and dist > REID_NO_APP_DIST:
            return None
        return (0.45 * (1.0 - iou(pred, box))
                + 0.25 * (dist / max_d)
                + 0.30 * (app if app is not None else 0.35))

    def _best_relink(self, frame_idx: int, p: dict, assigned: Dict[int, dict]) -> Optional[int]:
        best, best_cost = None, float("inf")
        for cid, c in self.customers.items():
            if cid in assigned or c.box is None:
                continue
            age = frame_idx - c.last_seen
            if age < 1 or age > REID_MAX_LOST_FRAMES:
                continue
            cost = self._relink_cost(frame_idx, c, p["box"], p["hist"], age)
            if cost is not None and cost < best_cost:
                best, best_cost = cid, cost
        return best

    @staticmethod
    def _det_hist(det: dict, frame: np.ndarray) -> Optional[np.ndarray]:
        if "hist" not in det:
            det["hist"] = appearance_hist(frame, det["box"])
        return det["hist"]

    def update(self, frame_idx: int, frame: np.ndarray, dets: List[dict]) -> Dict[int, dict]:
        """
        dets: [{"raw_id": int, "box": xyxy, "kpts": (17, 3)}, ...]
        Returns {customer_id: {"box", "kpts", "raw_id"}} for customers visible in this frame.
        """
        assigned: Dict[int, dict] = {}
        unknown: List[dict] = []
        for d in dets:
            cid = self.raw_to_cid.get(d["raw_id"])
            if cid is not None and cid in self.customers and cid not in assigned:
                assigned[cid] = d
            else:
                unknown.append(d)

        # Unknown raw tracks: confirm after a few frames, then re-link a lost customer or create a new one
        for d in unknown:
            p = self.pending.setdefault(d["raw_id"], {"n": 0, "hist": None, "last": frame_idx, "box": d["box"]})
            p["n"] += 1
            p["last"] = frame_idx
            p["box"] = d["box"]
            h = self._det_hist(d, frame)
            if h is not None:
                p["hist"] = h if p["hist"] is None else 0.5 * p["hist"] + 0.5 * h
        for raw in [r for r, p in self.pending.items() if frame_idx - p["last"] > PENDING_TTL]:
            del self.pending[raw]

        ready = sorted((d for d in unknown if self.pending[d["raw_id"]]["n"] >= CONFIRM_FRAMES),
                       key=lambda d: -self.pending[d["raw_id"]]["n"])
        for d in ready:
            raw = d["raw_id"]
            p = self.pending[raw]
            if any(iou(d["box"], a["box"]) > DUP_IOU for a in assigned.values()):
                continue                                   # duplicate detection of a known customer
            cid = self._best_relink(frame_idx, p, assigned)
            if cid is None:
                cid = self.next_id
                self.next_id += 1
                self.customers[cid] = Customer(cid=cid, first_seen=frame_idx, app_hist=p["hist"])
                print(f"[NEW CUSTOMER] Customer {cid} (tracker id {raw})")
            else:
                print(f"[RE-LINKED] Customer {cid} recovered as tracker id {raw} "
                      f"after {frame_idx - self.customers[cid].last_seen} frames")
            self._link(cid, raw)
            assigned[cid] = d
            del self.pending[raw]

        # Presence bookkeeping
        for cid, c in self.customers.items():
            if cid not in assigned:
                c.present = False
        for cid, d in assigned.items():
            c = self.customers[cid]
            if frame_idx - c.last_seen > WRIST_GRACE_FRAMES + 1:
                c.wrist_mem = [None, None]                 # stale short-term memory after a real absence
            c.present = True
            c.last_seen = frame_idx
            c.box = d["box"].copy()
            ctr = box_center(d["box"])
            c.center_hist.append((frame_idx, float(ctr[0]), float(ctr[1])))
            c.raw_id = d["raw_id"]

            # Learn the clothing signature slowly, never while overlapped (would absorb the occluder)
            overlapped = any(iou(d["box"], o["box"]) > APP_OVERLAP_IOU
                             for oc, o in assigned.items() if oc != cid)
            h = self._det_hist(d, frame)
            if h is not None and not overlapped:
                c.app_hist = h if c.app_hist is None else (1 - APP_UPDATE_ALPHA) * c.app_hist + APP_UPDATE_ALPHA * h

        return {cid: {"box": d["box"], "kpts": d["kpts"], "raw_id": d["raw_id"]} for cid, d in assigned.items()}

    def expire(self, frame_idx: int) -> None:
        """Forget customers unseen for too long (confirmed thieves are kept for the whole run)."""
        for cid in [c for c, s in self.customers.items()
                    if not s.present and not s.locked and frame_idx - s.last_seen > RETAIN_LOST_FRAMES]:
            raw = self.cid_to_raw.pop(cid, None)
            if raw is not None:
                self.raw_to_cid.pop(raw, None)
            del self.customers[cid]


# ----------------------------------------------------------------------------------------------
# Skeletal behaviour analysis + two-tier state machine
# ----------------------------------------------------------------------------------------------
def analyze_pose(c: Customer, kpts: np.ndarray, box: np.ndarray, frame_idx: int, params: Params) -> bool:
    """
    Update the customer's waist estimate, hand positions and reach history.
    Returns True when at least one hand is in the waist / torso zone in a concealment-like pose.
    """
    ARM_STRAIGHT_DEG = 140.0  # <--- Define it right here locally!
    
    h = box_height(box)
    cx_b = (float(box[0]) + float(box[2])) / 2.0
    ...

    # 1. Waist estimate with fallbacks, smoothed in box-relative coordinates
    center, torso, src = estimate_waist(kpts, box)
    rel = (float(center[0] - cx_b) / h, float(center[1] - float(box[1])) / h, torso / h)
    if c.waist_rel is None:
        c.waist_rel = rel
    else:
        c.waist_rel = tuple(WAIST_EMA * n + (1.0 - WAIST_EMA) * o for n, o in zip(rel, c.waist_rel))
    c.waist_src = src
    waist = np.array([cx_b + c.waist_rel[0] * h, float(box[1]) + c.waist_rel[1] * h], dtype=np.float32)
    torso = max(c.waist_rel[2] * h, 1.0)
    c.zone = (float(waist[0] - ZONE_HALF_W * torso), float(waist[1] + ZONE_TOP * torso),
              float(waist[0] + ZONE_HALF_W * torso), float(waist[1] + ZONE_BOTTOM * torso))

    # 2. Hands (fresh keypoint, or remembered for a few frames when the wrist drops out)
    any_ok = False
    hands: List[np.ndarray] = []
    for k, (w_i, e_i, s_i) in enumerate(ARMS):
        fresh = kpts[w_i][2] >= 0.45
        if fresh:
            wrist = kpts[w_i][:2].astype(np.float32)
            dx, dy = float((wrist[0] - waist[0]) / torso), float((wrist[1] - waist[1]) / torso)

            bent: Optional[bool] = None
            if kpts[e_i][2] >= 0.40 and kpts[s_i][2] >= 0.40:
                ARM_STRAIGHT_DEG = 140.0  # <--- Add this exact line right here!
                bent = True

            in_box = abs(dx) <= ZONE_HALF_W and ZONE_TOP <= dy <= ZONE_BOTTOM
            ok = in_box and (abs(dx) <= CENTER_DX or bent is True)
            c.wrist_mem[k] = (dx, dy, ok, 0)

            if dy <= REACH_UP or float(np.hypot(dx, dy)) >= REACH_DIST:
                c.last_reach = frame_idx
        else:
            mem = c.wrist_mem[k]
            if mem is not None and mem[3] < 2:  # Reduced grace frames so lost wrists expire faster
                dx, dy, ok, age = mem
                c.wrist_mem[k] = (dx, dy, ok, age + 1)
            else:
                c.wrist_mem[k] = None
                continue
        hands.append(waist + np.array([dx, dy], dtype=np.float32) * torso)
        any_ok = any_ok or ok
    c.hands = hands
    c.in_zone = any_ok
    return any_ok


def advance_state(c: Customer, active: bool, params: Params) -> Optional[str]:
    """
    Dwell timer + two-tier state machine. Returns "WARN" the frame a customer becomes SUSPICIOUS,
    "THEFT" the frame a theft is confirmed, otherwise None.
    """
    if active:
        c.dwell += 1
        c.gap = 0
    else:
        c.gap += 1
        if c.gap > DWELL_GAP_TOL:
            c.dwell = max(0, c.dwell - DWELL_DECAY)         # decays, never snaps to zero on a flicker

    if c.locked:
        c.state = "THEFT"
        return None

    if c.dwell > params.theft_frames:
        c.state, c.locked = "THEFT", True
        return "THEFT"
    if c.dwell > params.warn_frames:
        c.state = "SUSPICIOUS"
        if not c.warned:
            c.warned = True
            return "WARN"
        return None
    if c.state == "SUSPICIOUS" and c.dwell <= RELEASE_DWELL:
        c.state, c.warned = "BROWSING", False
    return None


# ----------------------------------------------------------------------------------------------
# Evidence export + logging
# ----------------------------------------------------------------------------------------------
def log_event(evidence_dir: Path, text: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  {text}"
    try:
        with open(evidence_dir / "events.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as e:
        print(f"[WARNING] Could not write events.log: {e}")


def export_theft_evidence(frame: np.ndarray, box: np.ndarray, cid: int, dwell: int,
                          evidence_dir: Path) -> List[Path]:
    """Full-scene snapshot (suspect boxed) + upscaled suspect crop, saved into evidence/."""
    saved: List[Path] = []
    try:
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(fw, int(box[2])), min(fh, int(box[3]))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        scene = frame.copy()
        cv2.rectangle(scene, (x1, y1), (x2, y2), COLOR_RED, 3)
        cv2.putText(scene, f"THEFT - Customer {cid} (dwell {dwell} frames)", (x1, max(y1 - 10, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
        scene_path = evidence_dir / f"theft_customer{cid}_{ts}_scene.jpg"
        if cv2.imwrite(str(scene_path), scene):
            saved.append(scene_path)

        pad_x, pad_y = int(0.12 * (x2 - x1)), int(0.08 * (y2 - y1))
        crop = frame[max(0, y1 - pad_y): min(fh, y2 + pad_y), max(0, x1 - pad_x): min(fw, x2 + pad_x)]
        if crop.shape[0] > 10 and crop.shape[1] > 10:
            if crop.shape[0] < 400:                          # 2x bicubic upscale for small CCTV crops
                crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            crop_path = evidence_dir / f"theft_customer{cid}_{ts}_suspect.jpg"
            if cv2.imwrite(str(crop_path), crop):
                saved.append(crop_path)
    except Exception as e:                                   # evidence export must never kill the pipeline
        print(f"[WARNING] Evidence export failure: {e}")
    return saved


# ----------------------------------------------------------------------------------------------
# HUD
# ----------------------------------------------------------------------------------------------
def draw_customer(img: np.ndarray, cid: int, pdata: dict, c: Customer, params: Params, frame_idx: int) -> None:
    box = pdata["box"].astype(int)
    kpts = pdata["kpts"]
    color = STATE_COLOR[c.state]

    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    for a, b in SKELETON:                                    # skeleton overlay
        if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
            cv2.line(img, tuple(kpts[a][:2].astype(int)), tuple(kpts[b][:2].astype(int)), color, 2)
    for i in range(17):
        if kpts[i][2] > 0.3:
            cv2.circle(img, tuple(kpts[i][:2].astype(int)), 3, (255, 255, 255), -1)

    if c.zone is not None:                                   # waist / torso zone
        cv2.rectangle(img, (int(c.zone[0]), int(c.zone[1])), (int(c.zone[2]), int(c.zone[3])),
                      COLOR_MAGENTA, 2 if c.in_zone else 1)
    for hpt in c.hands:
        cv2.circle(img, (int(hpt[0]), int(hpt[1])), 5, COLOR_MAGENTA, -1)

    label = f"Cust {cid} | {c.state} | {c.dwell}/{params.theft_frames}"
    cv2.putText(img, label, (box[0], max(box[1] - 10, 22 if cid % 2 == 0 else 42)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    engaged = "Y" if frame_idx - c.last_reach <= ENGAGE_WINDOW else "N"
    cv2.putText(img, f"{c.waist_src} z{int(c.in_zone)} r{engaged} t{c.raw_id}",
                (box[0], min(box[3] + 18, img.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_panel(img: np.ndarray, snapshot: Dict[int, dict]) -> None:
    y = 18
    for cid, d in sorted(snapshot.items()):
        cv2.putText(img, f"C{cid}: {d['state']:<10} dwell {d['dwell_time']:.1f}s ({d['dwell_frames']}f)",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 16


# ----------------------------------------------------------------------------------------------
# Tracker config
# ----------------------------------------------------------------------------------------------
def ensure_tracker_yaml(kind: str) -> str:
    """Write the tracker config next to this script and return its path (rewritten every run)."""
    out_dir = SCRIPT_DIR / "trackers"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{kind}_retail.yaml"
    lines = [f"# Auto-generated by run_inference.py ({kind}; long buffer for occlusions)"]
    for key, val in TRACKER_CFGS[kind].items():
        lines.append(f"{key}: {str(val).lower() if isinstance(val, bool) else val}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ----------------------------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Two-tier CCTV shoplifting detection (pose + dwell)")
    parser.add_argument("--source", type=str, default="", help="Video file path or camera index")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Person detection confidence")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference size (0 = model default)")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=sorted(TRACKER_CFGS.keys()))
    parser.add_argument("--tracker-cfg", type=str, default="", help="Custom tracker YAML (overrides the default)")
    parser.add_argument("--warn-frames", type=int, default=WARN_FRAMES, help="Tier 1: dwell frames -> SUSPICIOUS")
    parser.add_argument("--theft-frames", type=int, default=THEFT_FRAMES, help="Tier 2: dwell frames -> THEFT")
    parser.add_argument("--no-reach-gate", action="store_true",
                        help="Count waist dwell even without a preceding reach toward the shelf")
    parser.add_argument("--evidence-dir", type=str, default="evidence")
    parser.add_argument("--debug", action="store_true",
                        help="Open the HUD preview (boxes, skeletons, states, dwell counters, waist zone)")
    args = parser.parse_args(argv)

    if args.theft_frames <= args.warn_frames:
        parser.error("--theft-frames must be larger than --warn-frames")
    params = Params(warn_frames=args.warn_frames, theft_frames=args.theft_frames,
                    reach_gate=not args.no_reach_gate)

    video_source = args.source
    if not video_source:
        local_vids = sorted(Path(".").glob("*.mp4"))
        if not local_vids:
            raise FileNotFoundError("No --source given and no .mp4 found in the current directory.")
        video_source = str(local_vids[0])
        print(f"Auto-selected video source: {video_source}")
    if str(video_source).isdigit():
        video_source = int(video_source)

    model = YOLO(args.model)
    tracker_cfg = args.tracker_cfg or ensure_tracker_yaml(args.tracker)
    track_kwargs = dict(persist=True, tracker=tracker_cfg, verbose=False, conf=args.conf)
    if args.imgsz > 0:
        track_kwargs["imgsz"] = args.imgsz
    print(f"[INFO] Tracker: {args.tracker} ({tracker_cfg})")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    registry = CustomerRegistry(fps, params.theft_frames)
    show = args.debug
    if show and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # OpenCV's Qt backend aborts the whole process (no Python exception) when there is no display
        print("[WARNING] --debug requested but no display is available; running headless.")
        show = False
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Detect + track people (pose model + ByteTrack / BoT-SORT)
        res = model.track(frame, **track_kwargs)
        dets: List[dict] = []
        if (res and res[0].boxes is not None and res[0].boxes.id is not None
                and res[0].keypoints is not None):
            boxes = res[0].boxes.xyxy.cpu().numpy()
            raw_ids = res[0].boxes.id.int().cpu().numpy()
            keypoints = res[0].keypoints.data.cpu().numpy()
            for b, raw, k in zip(boxes, raw_ids, keypoints):
                box_w = b[2] - b[0]
                box_h = b[3] - b[1]
                # FILTER: Ignore distant background noise or distorted boxes
                if box_w < 35 or box_h < 70 or (box_h / max(box_w, 1)) < 1.2:
                    continue
                dets.append({"raw_id": int(raw), "box": b, "kpts": k})

        # 2. Stable customer IDs
        persons = registry.update(frame_idx, frame, dets)

        # 3. Per-customer skeletal analysis, dwell timer and tier transitions
        for cid, p in persons.items():
            c = registry.customers[cid]
            in_zone = analyze_pose(c, p["kpts"], p["box"], frame_idx, params)
            engaged = (not params.reach_gate) or (frame_idx - c.last_reach <= ENGAGE_WINDOW)
            
            # STICKY STATE LOCK: Once a customer hits the warning threshold, lock their state permanently!
            if not hasattr(c, 'locked_suspicious'):
                c.locked_suspicious = False
            
            if c.locked_suspicious or c.dwell >= params.warn_frames:
                c.locked_suspicious = True
                in_zone_effective = True  # Force state to stay active so it never decays back to green
            else:
                in_zone_effective = in_zone and engaged

            event = advance_state(c, in_zone_effective, params)
            
            if c.dwell >= params.warn_frames:
                c.locked_suspicious = True

            if event == "WARN":                                           # Tier 1
                msg = (f"[WARNING] Customer {cid} SUSPICIOUS: hands at waist/torso for {c.dwell} frames "
                       f"({c.dwell / fps:.1f}s) [waist source: {c.waist_src}]")
                print(msg)
                log_event(evidence_dir, msg)
            elif event == "THEFT":                                    # Tier 2
                c.theft_frame = frame_idx
                msg = (f"[THEFT ALERT] Customer {cid} CONFIRMED THEFT: sustained concealment gesture, "
                       f"dwell {c.dwell} frames ({c.dwell / fps:.1f}s)")
                print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)
                log_event(evidence_dir, msg)
                for path in export_theft_evidence(frame, p["box"], cid, c.dwell, evidence_dir):
                    print(f"[EVIDENCE] saved {path}")
                    log_event(evidence_dir, f"evidence saved: {path}")

        # 4. HUD preview
        if show:
            hud = frame.copy()
            for cid, p in persons.items():
                draw_customer(hud, cid, p, registry.customers[cid], params, frame_idx)
            draw_panel(hud, registry.snapshot(only_active=True))
            try:
                cv2.imshow("AI CCTV Shoplifting Detection", hud)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as e:
                print(f"[WARNING] HUD preview unavailable ({e}); continuing headless.")
                show = False

    cap.release()
    if args.debug:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print("\n[FINAL CUSTOMER STATES]")
    for cid, d in registry.snapshot(only_active=False).items():
        print(f"  Customer {cid}: {d}")


if __name__ == "__main__":
    main()