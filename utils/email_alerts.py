"""Non-blocking, cooldown-protected security email alerts."""

from __future__ import annotations

import smtplib
import threading
import time
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


class EmailAlertDispatcher:
    """Dispatch annotated detection screenshots through SMTP in background threads."""

    def __init__(
        self,
        sender_email: str,
        sender_password: str,
        receiver_email: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ) -> None:
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.alert_dir = Path(__file__).resolve().parents[1] / "alerts"
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self._last_alert: Dict[int, float] = {}
        self._lock = threading.Lock()

    def send_theft_alert(self, frame: np.ndarray, track_id: int, threat_score: float) -> Optional[Path]:
        """Queue an email alert if this track's five-second cooldown has elapsed.

        The frame is written synchronously so its data remains valid after video
        capture advances; SMTP communication happens in a daemon thread.
        """
        now = time.monotonic()
        track_id = int(track_id)
        with self._lock:
            if now - self._last_alert.get(track_id, float("-inf")) < 5.0:
                return None
            self._last_alert[track_id] = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.alert_dir / f"theft_ID{track_id}_{timestamp}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise IOError(f"Could not save alert frame to {image_path}")

        worker = threading.Thread(
            target=self._send_email,
            args=(image_path, track_id, float(threat_score)),
            daemon=True,
            name=f"shoplifting-email-{track_id}",
        )
        worker.start()
        return image_path

    def _send_email(self, image_path: Path, track_id: int, threat_score: float) -> None:
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = f"SECURITY ALERT: Shoplifting Detected (Track ID: {track_id})"
        message.attach(MIMEText(
            f"Potential shoplifting was detected for track ID {track_id}.\n"
            f"Threat probability: {threat_score:.2%}.\n"
            "The annotated camera frame is attached.",
            "plain",
        ))
        with image_path.open("rb") as image_file:
            attachment = MIMEApplication(image_file.read(), Name=image_path.name)
        attachment.add_header("Content-Disposition", "attachment", filename=image_path.name)
        message.attach(attachment)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, [self.receiver_email], message.as_string())
        except (OSError, smtplib.SMTPException) as exc:
            # Background workers must never interrupt live video inference.
            print(f"Email alert for track {track_id} could not be sent: {exc}")
