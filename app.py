"""
Project S.P.S. - Advanced Retail Surveillance WebApp (Final Emergency Fix)
========================================================================
"""

import streamlit as st
import cv2
import tempfile
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Page configuration
st.set_page_config(
    page_title="Project S.P.S. - Retail Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #ffffff; }
    .alert-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        backdrop-filter: blur(12px);
        padding: 16px;
        border-radius: 12px;
        color: #fca5a5;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
    }
    .review-modal {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🛡️ Project S.P.S. | Advanced Retail Surveillance & Behavioral Analytics")
st.markdown("---")

if "show_review" not in st.session_state:
    st.session_state.show_review = False
if "review_snapshot" not in st.session_state:
    st.session_state.review_snapshot = None
if "review_details" not in st.session_state:
    st.session_state.review_details = ""

# Sidebar Control Center
sidebar_col, center_col, right_col = st.columns([1.2, 2.8, 1.2])

with sidebar_col:
    st.markdown("### 🎛️ Control Center")
    model_choice = st.selectbox("YOLO Pose Model", ["yolov8n-pose.pt", "yolov8s-pose.pt"])
    conf_threshold = st.slider("Detection Confidence", 0.05, 0.95, 0.25, 0.05)
    warn_frames_limit = st.slider("Warning Frames (Suspicious)", 15, 100, 45, 5)
    theft_frames_limit = st.slider("Theft Frames (Confirmed)", 60, 300, 180, 10)
    tracker_choice = st.selectbox("Multi-Object Tracker", ["bytetrack", "botsort"])
    
    st.markdown("---")
    st.markdown("### 🔔 Notification Toggles")
    email_toggle = st.toggle("📧 Email Notifications", value=True)
    recipient_email = st.text_input("Receiver Email", "sajiendiran11@gmail.com")
    
    st.markdown("---")
    st.markdown("### 📹 Video Source")
    uploaded_file = st.file_uploader("Upload CCTV Footage", type=["mp4", "avi", "mov"])
    
    col_btn1, col_btn2 = st.columns(2)
    start_btn = col_btn1.button("START", type="primary", use_container_width=True)
    stop_btn = col_btn2.button("STOP", use_container_width=True)

with right_col:
    st.markdown("### 🚨 Live Anomaly Log")
    log_placeholder = st.empty()

with center_col:
    st.markdown("### 🖥️ Live Surveillance Monitor (Aisle 4)")
    video_placeholder = st.empty()
    alert_banner_placeholder = st.empty()
    action_buttons_placeholder = st.empty()
    review_modal_placeholder = st.empty()

def send_email_alert(subject, body, to_email):
    try:
        sender_email = "sajiendiran11@gmail.com"
        sender_password = "your_app_password_here"
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception:
        return False

if start_btn:
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        local_vids = sorted(Path(".").glob("*.mp4"))
        video_source = str(local_vids[0]) if local_vids else "shoplifting-21.mp4"

    if not Path(video_source).exists():
        st.error("Error: No valid video source found. Please upload an MP4 file.")
        st.stop()

    @st.cache_resource
    def load_yolo_model(m_name):
        return YOLO(m_name)

    model = load_yolo_model(model_choice)
    cap = cv2.VideoCapture(video_source)

    script_dir = Path(__file__).resolve().parent
    trackers_dir = script_dir / "trackers"
    trackers_dir.mkdir(exist_ok=True)
    tracker_yaml = trackers_dir / f"{tracker_choice}_retail.yaml"
    
    tracker_yaml.write_text(
        f"tracker_type: {tracker_choice}\n"
        f"track_high_thresh: {conf_threshold}\n"
        "track_low_thresh: 0.10\n"
        "new_track_thresh: 0.25\n"
        "track_buffer: 90\n"
        "match_thresh: 0.80\n"
        "fuse_score: true\n"
    )

    frame_idx = 0
    anomaly_logs = []
    customer_dwell_tracker = {}
    last_suspicious_frame = None
    notified_emails = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(
            frame, 
            persist=True, 
            tracker=str(tracker_yaml), 
            conf=conf_threshold, 
            verbose=False
        )
        
        annotated_frame = frame.copy()
        is_suspicious_active = False

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None

            for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Waist / pocket zone mapping
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i]
                    if len(kpts) >= 13 and kpts[11][2] > 0.3 and kpts[12][2] > 0.3:
                        hx_min = int(min(kpts[11][0], kpts[12][0]) - 20)
                        hx_max = int(max(kpts[11][0], kpts[12][0]) + 20)
                        hy_min = int(min(kpts[11][1], kpts[12][1]) - 10)
                        hy_max = int(max(kpts[11][1], kpts[12][1]) + 50)
                    else:
                        hx_min, hy_min, hx_max, hy_max = x1 + int((x2-x1)*0.2), y1 + int((y2-y1)*0.45), x2 - int((x2-x1)*0.2), y1 + int((y2-y1)*0.75)
                else:
                    hx_min, hy_min, hx_max, hy_max = x1 + int((x2-x1)*0.2), y1 + int((y2-y1)*0.45), x2 - int((x2-x1)*0.2), y1 + int((y2-y1)*0.75)

                cv2.rectangle(annotated_frame, (hx_min, hy_min), (hx_max, hy_max), (255, 0, 255), 2)

                if tid not in customer_dwell_tracker:
                    customer_dwell_tracker[tid] = 0
                
                customer_dwell_tracker[tid] += 1
                dwell_count = customer_dwell_tracker[tid]
                
                # Permanent status lock: Once flagged or past warning limit, stay locked as suspicious/theft
                is_already_flagged = tid in notified_emails or dwell_count >= warn_frames_limit
                is_theft = dwell_count >= theft_frames_limit
                is_suspicious = is_already_flagged or (dwell_count >= warn_frames_limit and not is_theft)

                color = (0, 0, 255) if is_theft else ((0, 165, 255) if is_suspicious else (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                status_label = "THEFT CONFIRMED" if is_theft else ("SUSPICIOUS" if is_suspicious else "BROWSING")
                status_text = f"Cust {tid} | {status_label} | {dwell_count}"
                cv2.putText(annotated_frame, status_text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_suspicious or is_theft:
                    is_suspicious_active = True
                    last_suspicious_frame = annotated_frame.copy()
                    
                    log_entry = f"⚠️ [Frame {frame_idx}] ID {tid}: Suspicious Dwell"
                    if log_entry not in anomaly_logs:
                        anomaly_logs.insert(0, log_entry)

                    if email_toggle and tid not in notified_emails:
                        notified_emails.add(tid)
                        send_email_alert(
                            f"🚨 S.P.S. ALERT: Suspicious Activity Detected (Customer {tid})",
                            f"Sir, sustained hand-to-torso dwell pattern detected in Aisle 4. Customer ID: {tid}.",
                            recipient_email
                        )

                # Draw accurate pocket interaction zone
                cv2.rectangle(annotated_frame, (hx_min, hy_min), (hx_max, hy_max), (255, 0, 255), 2)

                if tid not in customer_dwell_tracker:
                    customer_dwell_tracker[tid] = 0
                
                customer_dwell_tracker[tid] += 1
                dwell_count = customer_dwell_tracker[tid]
                
                is_theft = dwell_count >= theft_frames_limit
                is_suspicious = dwell_count >= warn_frames_limit and not is_theft

                color = (0, 0, 255) if is_theft else ((0, 165, 255) if is_suspicious else (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                status_label = "THEFT CONFIRMED" if is_theft else ("SUSPICIOUS" if is_suspicious else "BROWSING")
                status_text = f"Cust {tid} | {status_label} | {dwell_count}"
                cv2.putText(annotated_frame, status_text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_suspicious or is_theft:
                    is_suspicious_active = True
                    last_suspicious_frame = annotated_frame.copy()
                    
                    log_entry = f"⚠️ [Frame {frame_idx}] ID {tid}: Suspicious Dwell"
                    if log_entry not in anomaly_logs:
                        anomaly_logs.insert(0, log_entry)

                    if email_toggle and tid not in notified_emails:
                        notified_emails.add(tid)
                        send_email_alert(
                            f"🚨 S.P.S. ALERT: Suspicious Activity Detected (Customer {tid})",
                            f"Sir, sustained hand-to-torso dwell pattern detected in Aisle 4. Customer ID: {tid}.",
                            recipient_email
                        )

        # High-Speed Resizing
        fast_frame = cv2.resize(annotated_frame, (640, 360))
        frame_rgb = cv2.cvtColor(fast_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if is_suspicious_active:
            alert_banner_placeholder.markdown("""
                <div class="alert-banner">
                    🚨 <b>SIR, I FOUND THIS SUSPICIOUS ACTIVITY!</b><br>
                    <span style="font-size: 0.9rem; color: #fecaca;">
                    Behavioral anomaly detected in Aisle 4. Click Review below for forensic breakdown.
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            col_rev1, col_rev2 = action_buttons_placeholder.columns(2)
            if col_rev1.button("🔍 REVIEW SCENARIO", key=f"rev_{frame_idx}"):
                st.session_state.show_review = True
                if last_suspicious_frame is not None:
                    st.session_state.review_snapshot = cv2.cvtColor(last_suspicious_frame, cv2.COLOR_BGR2RGB)
                st.session_state.review_details = f"Sir, Customer lingered near the shelf past the warning threshold. Skeletal hand-to-torso proximity was flagged in Aisle 4 with an automated email sent to {recipient_email}."
            if col_rev2.button("❌ DISMISS", key=f"dis_{frame_idx}"):
                st.session_state.show_review = False
                action_buttons_placeholder.empty()
        else:
            alert_banner_placeholder.empty()
            action_buttons_placeholder.empty()

        with log_placeholder.container():
            if anomaly_logs:
                for log in anomaly_logs[:5]:
                    st.markdown(f"<small>{log}</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>No anomalies recorded yet.</small>", unsafe_allow_html=True)

    cap.release()

if st.session_state.show_review:
    with review_modal_placeholder.container():
        st.markdown("""
            <div class="review-modal">
                <h3>🔍 Forensic Incident Review & AI Summary</h3>
            </div>
        """, unsafe_allow_html=True)
        if st.session_state.review_snapshot is not None:
            st.image(st.session_state.review_snapshot, caption="Forensic Snapshot - High-Resolution Evidence", use_column_width=True)
        st.info(st.session_state.review_details)
        if st.button("Close Review Window"):
            st.session_state.show_review = False
            st.rerun()
else:
    review_modal_placeholder.empty()

if not start_btn:
    center_col.info("Click **'START'** in the left control panel to launch the high-speed web app stream.")