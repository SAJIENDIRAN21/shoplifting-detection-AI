"""
Real-time CCTV Shoplifting Prevention Inference Engine (Production Grade)
- Dual Model: YOLOv8-Pose + YOLOv8n Retail Detector
- Strict Pelvic Line Concealment Debounce (30 Frames)
- Complete Incident Capture: Pre-Roll (~3s) + Post-Roll (~4s) MP4 Video
- High-Resolution Forensic Image Dossier (Full Scene Context + Best Frame Profile)
- Real-Time Frame Playback
"""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO

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

# ----------------- Main Pipeline -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    video_source = args.source
    if not video_source:
        local_vids = list(Path(".").glob("*.mp4"))
        if local_vids:
            video_source = str(local_vids[0])
            print(f"Auto-selected video source: {video_source}")
        else:
            raise FileNotFoundError("No input video file found in current directory.")

    pose_model = YOLO("yolov8n-pose.pt")
    item_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)

    person_states = {}
    last_handoff = None
    rolling_buffer = deque(maxlen=60)  # ~3 seconds pre-roll

    # Post-roll incident capture states
    post_roll_active = False
    post_roll_frames_left = 0
    post_roll_buffer = []
    incident_ts = ""
    primary_tid_locked = None
    accomplice_tid_locked = None
    best_capture_frames = {}

    RETAIL_CLASSES = {"bottle", "cup", "cell phone"}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated_frame = frame.copy()
        rolling_buffer.append(frame.copy())

        # Post-roll continuous capture
        if post_roll_active:
            post_roll_buffer.append(frame.copy())
            post_roll_frames_left -= 1

        # 1. Track Persons via Pose
        pose_res = pose_model.track(frame, persist=True, verbose=False)
        current_persons = {}

        if pose_res and pose_res[0].boxes and pose_res[0].boxes.id is not None:
            boxes = pose_res[0].boxes.xyxy.cpu().numpy()
            track_ids = pose_res[0].boxes.id.int().cpu().numpy()
            keypoints = pose_res[0].keypoints.data.cpu().numpy()

            for b_xyxy, tid, kpts in zip(boxes, track_ids, keypoints):
                current_persons[tid] = {
                    "box": b_xyxy,
                    "kpts": kpts
                }
                if tid not in person_states:
                    person_states[tid] = {
                        "holding": False,
                        "item_box": None,
                        "disappeared": 0,
                        "role": "Normal",
                        "locked_theft": False
                    }

        # 2. Detect Retail Merchandise
        detected_items = []
        if frame_idx % 2 == 0:
            obj_res = item_model(frame, verbose=False, conf=args.conf)
            if obj_res and obj_res[0].boxes:
                for box in obj_res[0].boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = item_model.names[cls_id]
                    if cls_name in RETAIL_CLASSES:
                        xyxy = box.xyxy[0].cpu().numpy()
                        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        if bw < frame.shape[1] * 0.25 and bh < frame.shape[0] * 0.35:
                            detected_items.append({"class": cls_name, "box": xyxy})

        # 3. Associate Items to Wrist Keypoints
        for tid, pdata in current_persons.items():
            if person_states[tid]["locked_theft"]:
                continue

            kpts = pdata["kpts"]
            w_left, w_right = kpts[9][:2], kpts[10][:2]
            conf_l, conf_r = kpts[9][2], kpts[10][2]

            for item in detected_items:
                icx = (item["box"][0] + item["box"][2]) / 2.0
                icy = (item["box"][1] + item["box"][3]) / 2.0
                item_center = np.array([icx, icy])

                dl = np.linalg.norm(w_left - item_center) if conf_l > 0.35 else 999.0
                dr = np.linalg.norm(w_right - item_center) if conf_r > 0.35 else 999.0
                min_dist = min(dl, dr)

                if min_dist < 45.0:
                    person_states[tid]["holding"] = True
                    person_states[tid]["item_box"] = item["box"]
                    person_states[tid]["disappeared"] = 0

        # 4. Multi-Actor Coordinated Handoff
        for p1_id, p1_data in current_persons.items():
            if not person_states[p1_id]["holding"]:
                continue

            p1_wrists = [p1_data["kpts"][9][:2], p1_data["kpts"][10][:2]]
            p1_confs = [p1_data["kpts"][9][2], p1_data["kpts"][10][2]]

            for p2_id, p2_data in current_persons.items():
                if p1_id == p2_id:
                    continue

                p2_wrists = [p2_data["kpts"][9][:2], p2_data["kpts"][10][:2]]
                p2_confs = [p2_data["kpts"][9][2], p2_data["kpts"][10][2]]

                valid_dists = [
                    np.linalg.norm(w1 - w2)
                    for i, w1 in enumerate(p1_wrists)
                    for j, w2 in enumerate(p2_wrists)
                    if p1_confs[i] > 0.35 and p2_confs[j] > 0.35
                ]

                if valid_dists and min(valid_dists) < 55.0:
                    if last_handoff is None or (frame_idx - last_handoff[2]) > 30:
                        person_states[p2_id]["holding"] = True
                        person_states[p2_id]["item_box"] = person_states[p1_id]["item_box"]
                        person_states[p2_id]["disappeared"] = 0

                        person_states[p1_id]["holding"] = False
                        person_states[p1_id]["item_box"] = None
                        person_states[p1_id]["role"] = "Accomplice"
                        last_handoff = (p1_id, p2_id, frame_idx)

        # 5. Concealment Detection with Hip Gating & 30-Frame Counter
        for tid, pdata in current_persons.items():
            state = person_states[tid]
            if state["locked_theft"]:
                continue

            kpts = pdata["kpts"]
            box = pdata["box"]

            hip_confs = [kpts[11][2], kpts[12][2]]
            if max(hip_confs) > 0.35:
                hip_y = np.mean([kpts[i][1] for i in (11, 12) if kpts[i][2] > 0.35])
            else:
                hip_y = box[1] + (box[3] - box[1]) * 0.58

            visible_wrist_ys = [kpts[i][1] for i in (9, 10) if kpts[i][2] > 0.35]
            lowest_wrist_y = max(visible_wrist_ys) if visible_wrist_ys else 0.0

            if state["holding"]:
                item_visible = False
                for item in detected_items:
                    icx = (item["box"][0] + item["box"][2]) / 2.0
                    icy = (item["box"][1] + item["box"][3]) / 2.0

                    for w_idx in (9, 10):
                        if kpts[w_idx][2] > 0.35:
                            if np.linalg.norm(kpts[w_idx][:2] - [icx, icy]) < 55.0:
                                item_visible = True
                                state["item_box"] = item["box"]
                                state["disappeared"] = 0
                                break
                    if item_visible:
                        break

                if not item_visible:
                    state["disappeared"] += 1

                # Confirmed Concealment: Hand clearly down at waistband + gone for 30 frames (~1.5s)
                if lowest_wrist_y >= (hip_y - 10.0) and state["disappeared"] >= 30:
                    state["locked_theft"] = True
                    state["role"] = "PRIMARY THIEF: Concealment"
                    primary_tid_locked = tid

                    # Resolve Accomplice
                    accomplice_id = None
                    if last_handoff:
                        from_id, to_id, _ = last_handoff
                        if to_id == tid:
                            accomplice_id = from_id
                        elif from_id == tid:
                            accomplice_id = to_id

                    if accomplice_id and accomplice_id in person_states:
                        person_states[accomplice_id]["locked_theft"] = True
                        person_states[accomplice_id]["role"] = "ACCOMPLICE: Handoff"
                        accomplice_tid_locked = accomplice_id

                    # Initiate 80-frame (~4 sec) post-roll recording window
                    if not post_roll_active:
                        post_roll_active = True
                        post_roll_frames_left = 80
                        incident_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        post_roll_buffer = list(rolling_buffer)

        # 6. Suspect Photo Best-Frame Tracker (During Post-Roll)
        if post_roll_active:
            for s_id in [primary_tid_locked, accomplice_tid_locked]:
                if s_id and s_id in current_persons:
                    s_box = current_persons[s_id]["box"]
                    s_kpts = current_persons[s_id]["kpts"]
                    face_conf = np.mean([s_kpts[i][2] for i in range(5)])

                    if s_id not in best_capture_frames or face_conf > best_capture_frames[s_id]["conf"]:
                        best_capture_frames[s_id] = {
                            "frame": frame.copy(),
                            "box": s_box,
                            "conf": face_conf,
                            "role": person_states[s_id]["role"]
                        }

            # Finalize Incident Capture
            if post_roll_frames_left <= 0:
                post_roll_active = False

                # 1. Save Full Video Clip
                vid_path = evidence_dir / f"retail_theft_incident_{incident_ts}.mp4"
                threading.Thread(target=write_video_worker, args=(post_roll_buffer, vid_path, fps), daemon=True).start()

                # 2. Save Best Clear Profiles for Both Suspects
                for s_id, data in best_capture_frames.items():
                    threading.Thread(
                        target=write_dossier_worker,
                        args=(data["frame"], data["box"], s_id, data["role"], evidence_dir),
                        daemon=True
                    ).start()

        # 7. Production HUD Rendering
        for tid, pdata in current_persons.items():
            box = pdata["box"].astype(int)
            kpts = pdata["kpts"]
            state = person_states[tid]
            is_locked = state["locked_theft"]

            color = (0, 0, 255) if is_locked else (0, 255, 0)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            torso_pairs = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]
            for pa, pb in torso_pairs:
                if kpts[pa][2] > 0.3 and kpts[pb][2] > 0.3:
                    cv2.line(annotated_frame, tuple(kpts[pa][:2].astype(int)), tuple(kpts[pb][:2].astype(int)), color, 2)

            if state["holding"] and state["item_box"] is not None and not is_locked:
                ib = state["item_box"].astype(int)
                cv2.rectangle(annotated_frame, (ib[0], ib[1]), (ib[2], ib[3]), (255, 255, 0), 1)

            if is_locked:
                label = f"ID: {tid} | 100% THEFT: {state['role']}"
            else:
                label = f"ID: {tid} | Status: Browsing (Clear)"

            vertical_offset = 25 if (tid % 2 == 0) else 45
            label_y = max(box[1] - 10, vertical_offset)
            cv2.putText(annotated_frame, label, (box[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("AI CCTV Shoplifting Prevention", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()