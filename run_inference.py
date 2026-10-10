"""
Real-time CCTV Shoplifting Prevention Inference Engine (Production Grade)
- Dual Model: YOLOv8-Pose (ByteTrack) + YOLOv8n Retail Item Detector
- Bi-LSTM kinematic anomaly scoring (30-frame history), smoothed with a 10-frame EMA
- Skeletal-rotation gate: while the shoulders rotate, an invisible item is SELF-OCCLUSION, not concealment
- Concealment = verified hold + item gone + DOWNWARD wrist entry into the waist zone + >= 12-frame dwell
- Theft confirmation = concealment AND smoothed Bi-LSTM score > 0.70
- Proximity handoff (<= 70 px wrist contact for >= 3 frames) transfers the held item A -> B immediately
- Linked suspicion: whoever conceals is PRIMARY THIEF, the close-interaction partner is ACCOMPLICE
- Complete Incident Capture: Pre-Roll (~3s) + Post-Roll (~4s) MP4 Video
- High-Resolution Forensic Image Dossier (Full Scene Context + Best Frame Profile)
"""

import argparse
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ----------------- Tunable Constants -----------------
RETAIL_CLASSES = {"bottle", "cup", "cell phone"}

KP_CONF = 0.35            # Normal keypoint confidence gate
ZONE_KP_CONF = 0.20       # Looser gate for wrists near/inside pockets (pose conf drops when occluded)
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12

# --- Bi-LSTM ---
HISTORY_LEN = 30          # Bi-LSTM sequence length
SCORE_STRIDE = 2          # Run the Bi-LSTM every N frames (scores are cached in between)
EMA_SPAN_FRAMES = 10      # EMA span measured in video frames
EMA_ALPHA = 2.0 / (EMA_SPAN_FRAMES / SCORE_STRIDE + 1.0)   # per scoring update
AI_THEFT_SCORE = 0.70     # Smoothed score required (together with the physical dwell) for THEFT

# --- Item possession ---
HOLD_MIN_FRAMES = 8       # Sustained wrist/item proximity -> HOLDING
HOLD_GRACE_FRAMES = 10    # Detector dropout tolerated before an un-armed hold counter decays
MISSING_MIN_FRAMES = 5    # Item must be gone this long (outside rotation) before concealment counts
RELEASE_MISSING_FRAMES = 200  # Item gone this long (~10 s) with no concealment -> disarm
HAND_ITEM_FRAC = 0.10     # Wrist-to-item-box distance threshold as fraction of person height
HAND_ITEM_MIN_PX = 20.0

# --- Waist / pocket zone (in torso-length units, relative to hip midpoint) ---
ZONE_HALF_W = 0.75
ZONE_TOP = -0.25
ZONE_BOTTOM = 0.60

# --- Downward entry + dwell ---
DESC_WINDOW = 10          # Frames of wrist history inspected for a downward move
DESC_MIN_DROP = 0.12      # Wrist must have dropped at least this many torso-lengths into the zone
DESC_NO_RISE = 0.01       # ...and must not currently be moving back up
DWELL_FRAMES = 12         # Frames the wrist must stay in the zone after the downward entry
DWELL_GAP_TOL = 3         # Keypoint flicker tolerated inside a dwell

# --- Skeletal rotation (self-occlusion) ---
ROT_ENTER = 0.50          # shoulder aspect ratio below this => rotating (CLI: --rot-enter)
ROT_EXIT_MARGIN = 0.10    # hysteresis: must exceed ROT_ENTER + margin to start clearing
ROT_GRACE_FRAMES = 12     # keep treating item loss as self-occlusion this long after rotation ends
ROT_EMA = 0.5             # smoothing of the raw shoulder ratio

# --- Proximity handoff / linked suspicion ---
HANDOFF_PROX_PX = 70.0            # wrist-to-wrist distance (CLI: --prox-px)
HANDOFF_CONTACT_FRAMES = 3        # consecutive close frames (gap tolerant)
HANDOFF_CONTACT_GAP_TOL = 1
HANDOFF_WINDOW = 6                # loss must occur during contact or within this many frames after it
HANDOFF_LOST_MIN_FRAMES = 2       # giver's item must be undetected this long
HANDOFF_PRE_MARGIN = 10           # item must have been seen no earlier than this before contact began
HANDOFF_COOLDOWN = 30             # no repeat transfer on the same pair
INTERACTION_LINK_FRAMES = 300     # partner link stays valid this long (~15 s)

PRE_ROLL_FRAMES = 60
POST_ROLL_FRAMES = 80
STATE_TTL_FRAMES = 90             # Drop state of people not seen for this long

NEVER = -10 ** 9

# BGR colors
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)

FSM_STYLE = {
    "BROWSING": (COLOR_GREEN, "Browsing (Clear)"),
    "HOLDING": (COLOR_YELLOW, "Holding Item"),
    "SUSPICIOUS": (COLOR_ORANGE, "Suspicious Motion"),
    "THEFT": (COLOR_RED, ""),
}

TORSO_PAIRS = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]

NORM_MODES = ("hip", "hipbox", "box", "frame")


@dataclass
class Params:
    rot_enter: float = ROT_ENTER
    prox_px: float = HANDOFF_PROX_PX
    ai_gate: bool = True


# ----------------- Background Forensic Evidence Savers -----------------
def write_video_worker(frames, output_path, fps=20.0):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"\n[EVIDENCE LOGGED] Complete incident clip saved: {output_path.name}")


def write_dossier_worker(scene_frame, suspect_box, tid, role_name, evidence_dir):
    try:
        h, w = scene_frame.shape[:2]
        bx1, by1, bx2, by2 = suspect_box.astype(int)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        role_slug = "primary_thief" if "PRIMARY" in role_name else "accomplice"

        # 1. Full High-Res Scene Context (with red targeting rectangle)
        context_img = scene_frame.copy()
        cv2.rectangle(context_img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(
            context_img,
            f"SUSPECT ID: {tid} ({role_name})",
            (bx1, max(by1 - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        context_path = evidence_dir / f"suspect_{role_slug}_id_{tid}_{ts}_FULL_SCENE.jpg"
        cv2.imwrite(str(context_path), context_img)

        # 2. Enhanced Upper-Body / Profile Crop
        pad_x = int((bx2 - bx1) * 0.12)
        pad_y = int((by2 - by1) * 0.08)
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(w, bx2 + pad_x)
        cy2 = min(h, int(by1 + (by2 - by1) * 0.50))

        crop = scene_frame[cy1:cy2, cx1:cx2]
        if crop.shape[0] > 20 and crop.shape[1] > 20:
            # 2x Bicubic Upscale for Forensics
            crop_upscaled = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            profile_path = evidence_dir / f"suspect_{role_slug}_id_{tid}_{ts}_PROFILE.jpg"
            cv2.imwrite(str(profile_path), crop_upscaled)

        print(f"[DOSSIER LOGGED] High-res evidence stored for ID {tid} ({role_slug})")
    except Exception as e:
        print(f"[WARNING] Dossier export failure: {e}")


class IncidentRecorder:
    """Pre-roll + post-roll video capture and best-frame suspect dossiers."""

    def __init__(self, evidence_dir: Path, fps: float):
        self.evidence_dir = evidence_dir
        self.fps = fps
        self.active = False
        self.frames_left = 0
        self.buffer: List[np.ndarray] = []
        self.ts = ""
        self.suspects = set()
        self.best: Dict[int, dict] = {}

    def start(self, rolling_buffer):
        if self.active:
            return
        self.active = True
        self.frames_left = POST_ROLL_FRAMES
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.buffer = list(rolling_buffer)

    def add_suspect(self, tid: int):
        self.suspects.add(tid)

    def feed(self, frame):
        if self.active:
            self.buffer.append(frame.copy())
            self.frames_left -= 1

    def update(self, frame, persons, states):
        if not self.active:
            return
        for s_id in self.suspects:
            if s_id in persons:
                s_kpts = persons[s_id]["kpts"]
                face_conf = float(np.mean([s_kpts[i][2] for i in range(5)]))
                prev = self.best.get(s_id)
                if prev is None or face_conf > prev["conf"]:
                    self.best[s_id] = {
                        "frame": frame.copy(),
                        "box": persons[s_id]["box"].copy(),
                        "conf": face_conf,
                        "role": states[s_id].role,
                    }
        if self.frames_left <= 0:
            self.finish()

    def finish(self):
        if not self.active:
            return
        vid_path = self.evidence_dir / f"retail_theft_incident_{self.ts}.mp4"
        threading.Thread(
            target=write_video_worker, args=(self.buffer, vid_path, self.fps), daemon=True
        ).start()
        for s_id, data in self.best.items():
            threading.Thread(
                target=write_dossier_worker,
                args=(data["frame"], data["box"], s_id, data["role"], self.evidence_dir),
                daemon=True
            ).start()
        self.active = False
        self.frames_left = 0
        self.buffer = []
        self.suspects = set()
        self.best = {}


# ----------------- Geometry Helpers -----------------
def box_height(box: np.ndarray) -> float:
    return max(float(box[3] - box[1]), 1.0)


def torso_geometry(kpts: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, float]:
    """Hip midpoint and torso length (shoulder-mid to hip-mid), with box-based fallbacks."""
    box_h = box_height(box)

    hips = [kpts[i][:2] for i in (L_HIP, R_HIP) if kpts[i][2] > KP_CONF]
    if hips:
        hip_mid = np.mean(hips, axis=0)
    else:
        hip_mid = np.array([(box[0] + box[2]) / 2.0, box[1] + box_h * 0.58])

    shoulders = [kpts[i][:2] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] > KP_CONF]
    if shoulders:
        torso = float(np.linalg.norm(np.mean(shoulders, axis=0) - hip_mid))
    else:
        torso = 0.30 * box_h
    torso = max(torso, 0.15 * box_h)
    return hip_mid.astype(np.float32), torso


def normalize_keypoints(kpts: np.ndarray, box: np.ndarray, frame_wh: Tuple[int, int],
                        mode: str = "box") -> np.ndarray:
    """
    (17, 3) -> (51,) feature vector [x0, y0, c0, x1, y1, c1, ...].
    The (x, y) normalization must match what best_bilstm.pth was trained on:
      box    : (xy - box_top_left) / (box_w, box_h)          -> roughly [0, 1]   (default)
      hip    : (xy - hip_mid) / torso_length, clipped to [-5, 5]
      hipbox : (xy - hip_mid) / box_height
      frame  : xy / (frame_w, frame_h)                       -> [0, 1]
    Confidence is passed through unchanged.
    """
    xy_raw = kpts[:, :2]
    if mode == "frame":
        xy = xy_raw / np.array([max(frame_wh[0], 1), max(frame_wh[1], 1)], dtype=np.float32)
    elif mode == "box":
        bw = max(float(box[2] - box[0]), 1.0)
        bh = box_height(box)
        xy = (xy_raw - np.array([box[0], box[1]], dtype=np.float32)) / np.array([bw, bh], dtype=np.float32)
    elif mode == "hipbox":
        hip_mid, _ = torso_geometry(kpts, box)
        xy = (xy_raw - hip_mid) / box_height(box)
    else:  # "hip"
        hip_mid, torso = torso_geometry(kpts, box)
        xy = np.clip((xy_raw - hip_mid) / torso, -5.0, 5.0)
    out = np.concatenate([xy, kpts[:, 2:3]], axis=1)
    return out.reshape(-1).astype(np.float32)


def visible_wrists(kpts: np.ndarray, conf: float = KP_CONF) -> List[np.ndarray]:
    return [kpts[i][:2] for i in (L_WRIST, R_WRIST) if kpts[i][2] > conf]


def point_to_box_distance(p: np.ndarray, box: np.ndarray) -> float:
    dx = max(float(box[0] - p[0]), 0.0, float(p[0] - box[2]))
    dy = max(float(box[1] - p[1]), 0.0, float(p[1] - box[3]))
    return float(np.hypot(dx, dy))


def shoulder_aspect_ratio(kpts: np.ndarray, torso_h: float) -> Optional[float]:
    """|L_SHOULDER.x - R_SHOULDER.x| / torso_height. ~0.8 facing/away from camera, -> 0 when side-on."""
    if kpts[L_SHOULDER][2] > KP_CONF and kpts[R_SHOULDER][2] > KP_CONF:
        return abs(float(kpts[L_SHOULDER][0] - kpts[R_SHOULDER][0])) / max(torso_h, 1.0)
    return None


# ----------------- Bi-LSTM Anomaly Scorer -----------------
class AnomalyScorer:
    """
    Wraps BiLSTMPoseClassifier. If the model/weights are unavailable it returns no scores and the
    pipeline runs on the physical (descent + dwell) path alone.
    """

    def __init__(self, weights_path: Path, device: torch.device):
        self.device = device
        self.model = None
        self._load(weights_path)

    @property
    def using_model(self) -> bool:
        return self.model is not None

    def _load(self, weights_path: Path) -> None:
        try:
            from models.bilstm_classifier import BiLSTMPoseClassifier
        except Exception as e:
            print(f"[WARNING] Could not import BiLSTMPoseClassifier ({e}).")
            return

        if not weights_path.exists():
            print(f"[WARNING] Weights not found at '{weights_path}'.")
            return

        try:
            model = BiLSTMPoseClassifier()
            try:
                ckpt = torch.load(str(weights_path), map_location=self.device, weights_only=True)
            except TypeError:  # older torch without weights_only
                ckpt = torch.load(str(weights_path), map_location=self.device)

            if isinstance(ckpt, dict):
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        ckpt = ckpt[key]
                        break
                ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}

            model.load_state_dict(ckpt)
            model.to(self.device).eval()
            self.model = model
            print(f"[INFO] Bi-LSTM weights loaded: {weights_path}")
        except Exception as e:
            self.model = None
            print(f"[WARNING] Failed to load Bi-LSTM weights ({e}).")

    def score(self, sequences: Dict[int, np.ndarray]) -> Dict[int, float]:
        if not sequences or self.model is None:
            return {}
        ids = list(sequences.keys())
        batch = torch.from_numpy(np.stack([sequences[t] for t in ids])).float().to(self.device)
        with torch.inference_mode():
            out = self.model(batch).reshape(-1).cpu().numpy()
        out = np.nan_to_num(out, nan=0.0)
        return {tid: float(s) for tid, s in zip(ids, out)}


# ----------------- Per-Person State -----------------
@dataclass
class PersonState:
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    last_seen: int = 0

    # Item possession tracking
    hold_frames: int = 0
    lost_frames: int = 0                     # consecutive frames with no item at the hand
    missing_frames: int = 0                  # item gone while armed, NOT counting self-occluded frames
    rot_occluded_frames: int = 0             # item gone while rotating (treated as self-occlusion)
    armed: bool = False                      # Verified hold (>= HOLD_MIN_FRAMES) not yet released
    item_box: Optional[np.ndarray] = None
    last_item_frame: int = NEVER

    # Bi-LSTM (EMA-smoothed)
    score_raw: float = 0.0
    score: float = 0.0
    score_valid: bool = False

    # Skeletal rotation
    sh_ratio: float = 1.0
    rotating: bool = False
    rot_grace: int = 0

    # Wrist kinematics / dwell
    wrist_y: List[deque] = field(
        default_factory=lambda: [deque(maxlen=DESC_WINDOW), deque(maxlen=DESC_WINDOW)])
    in_zone: bool = False
    dwell_valid: bool = False                # a downward entry into the zone has been seen
    dwell: int = 0
    dwell_gap: int = 0

    # FSM
    fsm: str = "BROWSING"
    physical_ok: bool = False
    locked: bool = False
    role: str = "Normal"

    # Handoff / linked suspicion
    handoff_from: Optional[int] = None
    handoff_to: Optional[int] = None
    partner_id: Optional[int] = None
    partner_frame: int = NEVER

    def update_score(self, raw: float) -> None:
        self.score_raw = raw
        if not self.score_valid:
            self.score = raw
            self.score_valid = True
        else:
            self.score = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * self.score


@dataclass
class InteractionTrack:
    contact: int = 0
    gap: int = 0
    start_frame: int = NEVER
    last_ok: int = NEVER
    cooldown_until: int = NEVER
    last_transfer: int = NEVER


def reset_dwell(st: PersonState) -> None:
    st.dwell_valid = False
    st.dwell = 0
    st.dwell_gap = 0


# ----------------- Kinematics: rotation, downward entry, dwell -----------------
def update_kinematics(st: PersonState, kpts: np.ndarray, box: np.ndarray, params: Params) -> None:
    hip_mid, torso = torso_geometry(kpts, box)

    # 1. Skeletal rotation via smoothed shoulder aspect ratio (+ grace period after the turn)
    ratio = shoulder_aspect_ratio(kpts, torso)
    if ratio is not None:
        st.sh_ratio = ROT_EMA * ratio + (1.0 - ROT_EMA) * st.sh_ratio
    if st.sh_ratio < params.rot_enter:
        st.rot_grace = ROT_GRACE_FRAMES
    elif st.sh_ratio >= params.rot_enter + ROT_EXIT_MARGIN:
        st.rot_grace = max(0, st.rot_grace - 1)
    st.rotating = st.sh_ratio < params.rot_enter or st.rot_grace > 0

    # 2. Wrist position relative to the hip midpoint (torso units) -> zone membership + downward entry
    in_zone_any = False
    descent_event = False
    for k, idx in enumerate((L_WRIST, R_WRIST)):
        hist = st.wrist_y[k]
        if kpts[idx][2] >= ZONE_KP_CONF:
            y_rel = float((kpts[idx][1] - hip_mid[1]) / torso)
            x_rel = abs(float(kpts[idx][0] - hip_mid[0])) / torso
            in_zone = x_rel <= ZONE_HALF_W and ZONE_TOP <= y_rel <= ZONE_BOTTOM
        else:
            y_rel, in_zone = None, False
        hist.append(y_rel)

        if in_zone:
            in_zone_any = True
            vals = [v for v in hist if v is not None]
            recent = [v for v in list(hist)[-4:] if v is not None]
            if len(vals) >= 3 and len(recent) >= 2:
                dropped = y_rel - min(vals)                       # image y grows downward => dy > 0
                not_rising = (recent[-1] - recent[0]) >= -DESC_NO_RISE
                if dropped >= DESC_MIN_DROP and not_rising:
                    descent_event = True
    st.in_zone = in_zone_any

    # 3. Dwell: counts frames in the zone since a DOWNWARD entry. Never advances while rotating.
    if st.rotating:
        reset_dwell(st)
    elif in_zone_any:
        st.dwell_gap = 0
        if descent_event and not st.dwell_valid:
            st.dwell_valid = True
            st.dwell = 0
        if st.dwell_valid:
            st.dwell += 1
    elif st.dwell_valid:
        st.dwell_gap += 1
        if st.dwell_gap > DWELL_GAP_TOL:
            reset_dwell(st)


# ----------------- Detection / Association -----------------
def detect_items(item_model, frame: np.ndarray, conf: float) -> List[dict]:
    items: List[dict] = []
    res = item_model(frame, verbose=False, conf=conf)
    if not res or res[0].boxes is None:
        return items
    fh, fw = frame.shape[:2]
    for box in res[0].boxes:
        cls_name = item_model.names[int(box.cls[0].item())]
        if cls_name not in RETAIL_CLASSES:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        if bw < fw * 0.25 and bh < fh * 0.35:
            items.append({"class": cls_name, "box": xyxy})
    return items


def associate_items(items: List[dict], persons: Dict[int, dict]) -> Dict[int, Optional[int]]:
    """
    item_near[tid] -> index of the item owned by that person's hand (or None). Each item is owned by
    at most one person: the one whose wrist is closest to it.
    """
    dist: Dict[int, List[float]] = {}
    thr: Dict[int, float] = {}
    for tid, p in persons.items():
        thr[tid] = max(HAND_ITEM_MIN_PX, HAND_ITEM_FRAC * box_height(p["box"]))
        wrists = visible_wrists(p["kpts"])
        dist[tid] = [
            min(point_to_box_distance(w, it["box"]) for w in wrists) if wrists else float("inf")
            for it in items
        ]

    item_near: Dict[int, Optional[int]] = {tid: None for tid in persons}
    best_d: Dict[int, float] = {}
    for i in range(len(items)):
        cands = [(dist[tid][i], tid) for tid in persons if dist[tid][i] <= thr[tid]]
        if not cands:
            continue
        d, owner = min(cands)
        if item_near[owner] is None or d < best_d[owner]:
            item_near[owner] = i
            best_d[owner] = d
    return item_near


def update_item_tracking(st: PersonState, item_idx: Optional[int], items: List[dict], frame_idx: int) -> None:
    if item_idx is not None:
        st.item_box = items[item_idx]["box"]
        st.hold_frames += 1
        st.lost_frames = 0
        st.missing_frames = 0
        st.rot_occluded_frames = 0
        st.last_item_frame = frame_idx
        reset_dwell(st)                      # item visible again -> any pending concealment is void
        if st.hold_frames >= HOLD_MIN_FRAMES:
            st.armed = True
        return

    st.lost_frames += 1
    if st.armed:
        if st.rotating:
            st.rot_occluded_frames += 1      # self-occlusion: state is held, not escalated
        else:
            st.missing_frames += 1
    elif st.lost_frames > HOLD_GRACE_FRAMES:
        st.hold_frames = 0
        st.item_box = None


# ----------------- Finite State Machine -----------------
def evaluate_fsm(st: PersonState, ai_gate: bool) -> bool:
    """Advance the FSM for one person. Returns True on the frame THEFT is first confirmed."""
    if st.locked:
        st.fsm = "THEFT"
        return False

    item_gone = st.armed and st.missing_frames >= MISSING_MIN_FRAMES
    st.physical_ok = (
        item_gone
        and not st.rotating
        and st.dwell_valid
        and st.dwell >= DWELL_FRAMES
    )
    ai_ok = (not ai_gate) or (st.score_valid and st.score > AI_THEFT_SCORE)

    # Item gone for a very long time with no concealment -> put down / lost by the detector
    if st.armed and st.missing_frames >= RELEASE_MISSING_FRAMES and not st.physical_ok:
        st.armed = False
        st.hold_frames = 0
        st.missing_frames = 0
        st.item_box = None
        reset_dwell(st)

    if st.physical_ok and ai_ok:
        st.fsm = "THEFT"
        return True
    if st.physical_ok:
        st.fsm = "SUSPICIOUS"
    elif st.armed:
        st.fsm = "HOLDING"
    else:
        st.fsm = "BROWSING"
    return False


def lock_theft(tid: int, states: Dict[int, PersonState], recorder: IncidentRecorder,
               rolling_buffer, frame_idx: int) -> None:
    st = states[tid]
    st.locked = True
    st.fsm = "THEFT"
    st.role = "PRIMARY THIEF: Concealment"
    recorder.start(rolling_buffer)
    recorder.add_suspect(tid)
    print(f"\n[THEFT CONFIRMED] ID {tid} (smoothed Bi-LSTM {st.score:.2f}, dwell {st.dwell} frames)")

    # Linked suspicion: whoever interacted closely with the concealer while an item was held.
    pid = st.partner_id
    if (pid is not None and pid in states and not states[pid].locked
            and frame_idx - st.partner_frame <= INTERACTION_LINK_FRAMES):
        acc = states[pid]
        acc.locked = True
        acc.fsm = "THEFT"
        acc.role = "ACCOMPLICE: Handoff"
        recorder.add_suspect(pid)
        print(f"[ACCOMPLICE CONFIRMED] ID {pid} (close interaction with ID {tid})")


# ----------------- Proximity Handoff + Linked Suspicion -----------------
def wrists_within(kpts_a: np.ndarray, kpts_b: np.ndarray, max_dist: float) -> bool:
    wa = visible_wrists(kpts_a, ZONE_KP_CONF)
    wb = visible_wrists(kpts_b, ZONE_KP_CONF)
    return any(float(np.linalg.norm(a - b)) <= max_dist for a in wa for b in wb)


def try_transfer(frame_idx: int, giver_id: int, taker_id: int, states: Dict[int, PersonState],
                 tr: InteractionTrack, handoff_log: list) -> bool:
    """Move the hold from giver to taker if the giver just lost the item during a close contact."""
    g, t = states[giver_id], states[taker_id]
    if g.locked or t.locked or not g.armed:
        return False
    if g.lost_frames < HANDOFF_LOST_MIN_FRAMES:
        return False                                        # giver still visibly holds it
    if g.last_item_frame < tr.start_frame - HANDOFF_PRE_MARGIN:
        return False                                        # item vanished long before this contact
    if g.last_item_frame <= tr.last_transfer:
        return False                                        # item not seen since the last transfer: no ping-pong

    # Immediate transfer of the hold to the taker (the pass itself is usually hidden by the hands)
    already_visible = t.armed and t.lost_frames == 0
    t.armed = True
    t.hold_frames = max(t.hold_frames, HOLD_MIN_FRAMES)
    if not already_visible:
        t.lost_frames = g.lost_frames
        t.missing_frames = 0
        t.rot_occluded_frames = 0
        t.item_box = g.item_box
        t.last_item_frame = g.last_item_frame
    reset_dwell(t)                                          # concealment must come AFTER the transfer
    t.handoff_from = giver_id

    g.armed = False
    g.hold_frames = 0
    g.missing_frames = 0
    g.rot_occluded_frames = 0
    g.item_box = None
    reset_dwell(g)
    g.handoff_to = taker_id

    for a, b, when in ((g, taker_id, frame_idx), (t, giver_id, frame_idx)):
        a.partner_id, a.partner_frame = b, when

    tr.cooldown_until = frame_idx + HANDOFF_COOLDOWN
    tr.last_transfer = frame_idx
    tr.last_ok = NEVER
    tr.contact = 0            # a new transfer needs fresh contact AND an item seen after it began
    tr.gap = 0
    handoff_log.append((giver_id, taker_id, frame_idx))
    print(f"\n[HANDOFF] ID {giver_id} -> ID {taker_id} (proximity transfer, item hidden by hands)")
    return True


def process_interactions(frame_idx: int, persons: Dict[int, dict], states: Dict[int, PersonState],
                         tracks: Dict[Tuple[int, int], InteractionTrack], params: Params,
                         handoff_log: list) -> None:
    ids = sorted(persons.keys())
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            sa, sb = states[a_id], states[b_id]
            tr = tracks.setdefault((a_id, b_id), InteractionTrack())

            relevant = (sa.armed and not sa.locked) or (sb.armed and not sb.locked)
            close = relevant and wrists_within(persons[a_id]["kpts"], persons[b_id]["kpts"], params.prox_px)

            if close:
                if tr.contact == 0:
                    tr.start_frame = frame_idx
                tr.contact += 1
                tr.gap = 0
            else:
                tr.gap += 1
                if tr.gap > HANDOFF_CONTACT_GAP_TOL:
                    tr.contact = 0

            if tr.contact >= HANDOFF_CONTACT_FRAMES:
                tr.last_ok = frame_idx
                # Linked suspicion: remember who interacted with whom while an item was held
                sa.partner_id, sa.partner_frame = b_id, frame_idx
                sb.partner_id, sb.partner_frame = a_id, frame_idx

            if frame_idx < tr.cooldown_until or frame_idx - tr.last_ok > HANDOFF_WINDOW:
                continue
            if not try_transfer(frame_idx, a_id, b_id, states, tr, handoff_log):
                try_transfer(frame_idx, b_id, a_id, states, tr, handoff_log)

    # Drop stale tracks (a person left the scene)
    for key in [k for k in tracks if k[0] not in persons or k[1] not in persons]:
        del tracks[key]


# ----------------- HUD -----------------
def draw_person(annotated: np.ndarray, tid: int, pdata: dict, st: PersonState, debug: bool) -> None:
    box = pdata["box"].astype(int)
    kpts = pdata["kpts"]
    color, text = FSM_STYLE[st.fsm]

    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

    for pa, pb in TORSO_PAIRS:
        if kpts[pa][2] > 0.3 and kpts[pb][2] > 0.3:
            cv2.line(annotated, tuple(kpts[pa][:2].astype(int)), tuple(kpts[pb][:2].astype(int)), color, 2)

    if st.armed and st.item_box is not None and st.lost_frames <= HOLD_GRACE_FRAMES and not st.locked:
        ib = st.item_box.astype(int)
        cv2.rectangle(annotated, (ib[0], ib[1]), (ib[2], ib[3]), (255, 255, 0), 1)

    if st.fsm == "THEFT":
        label = f"ID: {tid} | 100% THEFT: {st.role}"
    else:
        label = f"ID: {tid} | Status: {text}"
        if st.armed and st.rotating and st.lost_frames > 0:
            label += " (turning: self-occluded)"
        if st.handoff_to is not None:
            label += f" | gave item to {st.handoff_to}"
        if st.handoff_from is not None:
            label += f" | got item from {st.handoff_from}"
    if st.score_valid:
        label += f" | AI {st.score:.2f}"

    vertical_offset = 25 if (tid % 2 == 0) else 45
    label_y = max(box[1] - 10, vertical_offset)
    cv2.putText(annotated, label, (box[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if debug:
        hip_mid, torso = torso_geometry(kpts, pdata["box"])
        zx1 = int(hip_mid[0] - ZONE_HALF_W * torso)
        zx2 = int(hip_mid[0] + ZONE_HALF_W * torso)
        zy1 = int(hip_mid[1] + ZONE_TOP * torso)
        zy2 = int(hip_mid[1] + ZONE_BOTTOM * torso)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 0, 255), 1)
        dbg = (f"hold:{st.hold_frames} miss:{st.missing_frames} rotocc:{st.rot_occluded_frames} "
               f"dwell:{st.dwell}/{DWELL_FRAMES}{'*' if st.dwell_valid else ''} "
               f"sh:{st.sh_ratio:.2f}{' ROT' if st.rotating else ''} zone:{int(st.in_zone)}")
        cv2.putText(annotated, dbg, (box[0], min(box[3] + 18, annotated.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ----------------- Main Pipeline -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.35, help="Item detector confidence")
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--item-model", type=str, default="yolov8n.pt")
    parser.add_argument("--weights", type=str, default="models/best_bilstm.pth")
    parser.add_argument("--norm", type=str, default="box", choices=NORM_MODES,
                        help="Keypoint normalization fed to the Bi-LSTM (must match training)")
    parser.add_argument("--rot-enter", type=float, default=ROT_ENTER,
                        help="Shoulder aspect ratio below which the person counts as rotating")
    parser.add_argument("--prox-px", type=float, default=HANDOFF_PROX_PX,
                        help="Wrist-to-wrist distance (pixels) for a proximity handoff")
    parser.add_argument("--no-ai-gate", action="store_true",
                        help="Debug: confirm theft on the physical dwell alone (ignore the Bi-LSTM score)")
    parser.add_argument("--item-stride", type=int, default=2, help="Run item detector every N frames")
    parser.add_argument("--no-show", action="store_true", help="Disable the preview window")
    parser.add_argument("--debug", action="store_true", help="Draw waist zone and FSM counters")
    args = parser.parse_args()

    video_source = args.source
    if not video_source:
        local_vids = list(Path(".").glob("*.mp4"))
        if local_vids:
            video_source = str(local_vids[0])
            print(f"Auto-selected video source: {video_source}")
        else:
            raise FileNotFoundError("No input video file found in current directory.")
    if video_source.isdigit():
        video_source = int(video_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = YOLO(args.pose_model)
    item_model = YOLO(args.item_model)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute() and not weights_path.exists():
        weights_path = SCRIPT_DIR / weights_path
    scorer = AnomalyScorer(weights_path, device)

    ai_gate = scorer.using_model and not args.no_ai_gate
    if not scorer.using_model:
        print("[WARNING] No Bi-LSTM available: AI gate disabled, theft is confirmed on the physical dwell alone.")
    elif args.no_ai_gate:
        print("[INFO] --no-ai-gate: theft is confirmed on the physical dwell alone.")
    params = Params(rot_enter=args.rot_enter, prox_px=args.prox_px, ai_gate=ai_gate)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)

    person_states: Dict[int, PersonState] = {}
    interaction_tracks: Dict[Tuple[int, int], InteractionTrack] = {}
    handoff_log: List[Tuple[int, int, int]] = []
    rolling_buffer = deque(maxlen=PRE_ROLL_FRAMES)  # ~3 seconds pre-roll
    recorder = IncidentRecorder(evidence_dir, fps)

    detected_items: List[dict] = []
    frame_idx = 0
    if ai_gate:
        scorer_tag = f"Bi-LSTM[{args.norm}] EMA>{AI_THEFT_SCORE:.2f} + dwell"
    elif scorer.using_model:
        scorer_tag = f"Bi-LSTM[{args.norm}] (gate OFF) dwell only"
    else:
        scorer_tag = "Dwell only (no weights)"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_h, frame_w = frame.shape[:2]
        annotated_frame = frame.copy()
        rolling_buffer.append(frame.copy())
        recorder.feed(frame)

        # 1. Track persons via pose (ByteTrack); update keypoint history, rotation, wrist kinematics
        pose_res = pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_persons: Dict[int, dict] = {}

        if (pose_res and pose_res[0].boxes is not None and pose_res[0].boxes.id is not None
                and pose_res[0].keypoints is not None):
            boxes = pose_res[0].boxes.xyxy.cpu().numpy()
            track_ids = pose_res[0].boxes.id.int().cpu().numpy()
            keypoints = pose_res[0].keypoints.data.cpu().numpy()

            for b_xyxy, raw_tid, kpts in zip(boxes, track_ids, keypoints):
                tid = int(raw_tid)
                current_persons[tid] = {"box": b_xyxy, "kpts": kpts}
                st = person_states.setdefault(tid, PersonState())
                st.last_seen = frame_idx
                st.history.append(normalize_keypoints(kpts, b_xyxy, (frame_w, frame_h), args.norm))
                if not st.locked:
                    update_kinematics(st, kpts, b_xyxy, params)

        # 2. Detect retail merchandise (cached between detector runs)
        if (frame_idx - 1) % max(1, args.item_stride) == 0:
            detected_items = detect_items(item_model, frame, args.conf)

        # 3. Associate items to wrists and update possession counters (rotation-aware)
        item_near = associate_items(detected_items, current_persons)
        for tid in current_persons:
            st = person_states[tid]
            if st.locked:
                continue
            update_item_tracking(st, item_near.get(tid), detected_items, frame_idx)

        # 4. Proximity handoff + linked-suspicion bookkeeping
        process_interactions(frame_idx, current_persons, person_states, interaction_tracks,
                             params, handoff_log)

        # 5. Bi-LSTM anomaly scoring (batched, every SCORE_STRIDE frames) with EMA smoothing
        if scorer.using_model and frame_idx % SCORE_STRIDE == 0:
            sequences = {
                tid: np.stack(person_states[tid].history)
                for tid in current_persons
                if len(person_states[tid].history) == HISTORY_LEN
            }
            for tid, s in scorer.score(sequences).items():
                person_states[tid].update_score(s)

        # 6. FSM evaluation + theft lock (+ accomplice linking)
        for tid in current_persons:
            if evaluate_fsm(person_states[tid], ai_gate):
                lock_theft(tid, person_states, recorder, rolling_buffer, frame_idx)

        # 7. Evidence capture (best-frame tracker + finalize)
        recorder.update(frame, current_persons, person_states)

        # 8. Stale state cleanup
        for tid in [t for t, s in person_states.items() if frame_idx - s.last_seen > STATE_TTL_FRAMES]:
            if not person_states[tid].locked:
                del person_states[tid]

        # 9. HUD rendering
        for tid, pdata in current_persons.items():
            draw_person(annotated_frame, tid, pdata, person_states[tid], args.debug)
        cv2.putText(annotated_frame, f"Scorer: {scorer_tag}", (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if not args.no_show:
            cv2.imshow("AI CCTV Shoplifting Prevention", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Flush an in-progress incident if the video ended during post-roll
    recorder.finish()

    cap.release()
    cv2.destroyAllWindows()
    # Give background writer threads a moment to flush before interpreter exit
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=10.0)


if __name__ == "__main__":
    main()