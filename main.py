import sys
import cv2
import ollama
import tempfile
import os
import re
import json
import time

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QSplitter,
    QSizePolicy
)

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "gemma4:31b-cloud"

CAMERA_INDEX = 0

STABLE_TIME = 1.5

SCAN_COOLDOWN = 5.0

GUIDE_WIDTH_RATIO = 0.78
GUIDE_HEIGHT_RATIO = 0.55

MIN_EDGE_DENSITY = 0.025
MIN_VARIANCE = 180.0


# ============================================================
# CLEAN AI OUTPUT
# ============================================================

def clean_output(text):

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    text = text.replace("**", "")
    text = text.replace("*", "")

    return text.strip()


# ============================================================
# PARSE JSON
# ============================================================

def parse_json_response(text):

    cleaned = clean_output(text)

    try:

        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        re.DOTALL
    )

    if match:

        try:

            data = json.loads(
                match.group(0)
            )

            if isinstance(data, dict):
                return data

        except Exception:
            pass

    return None


# ============================================================
# PROFESSIONAL DISPLAY
# ============================================================

def format_professional(data):

    full_name = data.get(
        "FullName",
        "N/A"
    )

    id_number = data.get(
        "IDNumber",
        "N/A"
    )

    nationality = data.get(
        "Nationality",
        "N/A"
    )

    dob = data.get(
        "DateOfBirth",
        "N/A"
    )

    issuing_date = data.get(
        "IssuingDate",
        "N/A"
    )

    expiry_date = data.get(
        "ExpiryDate",
        "N/A"
    )

    return (
        "╔══════════════════════════════════════════════╗\n"
        "║              EMIRATES ID DETAILS             ║\n"
        "╚══════════════════════════════════════════════╝\n\n"

        f"  Full Name       : {full_name}\n\n"

        f"  ID Number       : {id_number}\n\n"

        f"  Nationality     : {nationality}\n\n"

        f"  Date of Birth   : {dob}\n\n"

        f"  Issuing Date    : {issuing_date}\n\n"

        f"  Expiry Date     : {expiry_date}\n\n"

        "──────────────────────────────────────────────\n"
        "              EXTRACTION COMPLETE\n"
        "──────────────────────────────────────────────"
    )


# ============================================================
# GUIDE RECTANGLE
# ============================================================

def get_guide(frame):

    h, w = frame.shape[:2]

    guide_w = int(
        w * GUIDE_WIDTH_RATIO
    )

    guide_h = int(
        h * GUIDE_HEIGHT_RATIO
    )

    x1 = int(
        (w - guide_w) / 2
    )

    y1 = int(
        (h - guide_h) / 2
    )

    x2 = x1 + guide_w
    y2 = y1 + guide_h

    return x1, y1, x2, y2


# ============================================================
# ID DETECTION
# ============================================================

def detect_id_in_guide(frame):

    x1, y1, x2, y2 = get_guide(
        frame
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:

        return False, 0, 0

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    edge_pixels = cv2.countNonZero(
        edges
    )

    total_pixels = (
        edges.shape[0]
        * edges.shape[1]
    )

    edge_density = (
        edge_pixels
        / float(total_pixels)
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_area = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area > largest_area:
            largest_area = area

    roi_area = (
        roi.shape[0]
        * roi.shape[1]
    )

    contour_ratio = (
        largest_area
        / float(roi_area)
    )

    detected = False

    if (
        edge_density >= MIN_EDGE_DENSITY
        and variance >= MIN_VARIANCE
    ):

        detected = True

    elif contour_ratio > 0.15:

        detected = True

    return (
        detected,
        edge_density,
        variance
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class IDCaptureApp(QWidget):

    def __init__(self):

        super().__init__()

        # ====================================================
        # WINDOW
        # ====================================================

        self.setWindowTitle(
            "AI Automatic Emirates ID Scanner"
        )

        self.resize(
            1200,
            900
        )

        # ====================================================
        # CAMERA LABEL
        # ====================================================

        self.camera_label = QLabel()

        self.camera_label.setMinimumSize(
            500,
            300
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #020617;
                border: 2px solid #334155;
                border-radius: 10px;
                color: #94A3B8;
                font-size: 20px;
            }
        """)

        self.camera_label.setText(
            "Starting camera..."
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = QLabel(
            "● Place Emirates ID inside the guide"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setMinimumHeight(
            35
        )

        self.status_label.setStyleSheet("""
            QLabel {
                color: #60A5FA;
                font-size: 16px;
                font-weight: bold;
            }
        """)

        # ====================================================
        # OUTPUT AREA
        # ====================================================

        self.output_area = QTextEdit()

        self.output_area.setReadOnly(
            True
        )

        self.output_area.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.output_area.setPlaceholderText(
            "Automatically extracted Emirates ID information will appear here..."
        )

        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 2px solid #334155;
                border-radius: 10px;
                padding: 15px;
                font-family: Consolas;
                font-size: 16px;
            }
        """)

        # ====================================================
        # SAVE BUTTON
        # ====================================================

        self.save_btn = QPushButton(
            "SAVE EXTRACTED DATA"
        )

        self.save_btn.setMinimumHeight(
            45
        )

        self.save_btn.setEnabled(
            False
        )

        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-size: 15px;
                font-weight: bold;
                border-radius: 7px;
                padding: 10px;
            }

            QPushButton:hover {
                background-color: #047857;
            }

            QPushButton:disabled {
                background-color: #475569;
                color: #94A3B8;
            }
        """)

        self.save_btn.clicked.connect(
            self.save_to_file
        )

        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_panel = QWidget()

        camera_layout = QVBoxLayout(
            camera_panel
        )

        camera_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        camera_layout.addWidget(
            self.camera_label
        )

        camera_layout.addWidget(
            self.status_label
        )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        info_panel = QWidget()

        info_layout = QVBoxLayout(
            info_panel
        )

        info_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        info_layout.addWidget(
            self.output_area
        )

        info_layout.addWidget(
            self.save_btn
        )

        # ====================================================
        # RESIZABLE SPLITTER
        # ====================================================

        self.splitter = QSplitter(
            Qt.Vertical
        )

        self.splitter.addWidget(
            camera_panel
        )

        self.splitter.addWidget(
            info_panel
        )

        # Initial sizes
        self.splitter.setSizes(
            [580, 300]
        )

        # Allow both areas to resize
        self.splitter.setStretchFactor(
            0,
            3
        )

        self.splitter.setStretchFactor(
            1,
            2
        )

        self.splitter.setHandleWidth(
            8
        )

        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #334155;
                height: 8px;
            }

            QSplitter::handle:hover {
                background-color: #3B82F6;
            }
        """)

        # ====================================================
        # MAIN LAYOUT
        # ====================================================

        main_layout = QVBoxLayout()

        main_layout.setContentsMargins(
            10,
            10,
            10,
            10
        )

        main_layout.addWidget(
            self.splitter
        )

        self.setLayout(
            main_layout
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        if not self.cap.isOpened():

            self.status_label.setText(
                "● ERROR: Camera not available"
            )

            self.status_label.setStyleSheet("""
                QLabel {
                    color: #EF4444;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

        else:

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

        # ====================================================
        # TIMER
        # ====================================================

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(
            30
        )

        # ====================================================
        # STATE
        # ====================================================

        self.card_detected = False

        self.stable_start = None

        self.processing = False

        self.last_scan_time = 0

        self.last_parsed = None

        self.stable_frames = 0

    # ========================================================
    # CAMERA UPDATE
    # ========================================================

    def update_frame(self):

        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        preview = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # GUIDE
        # ----------------------------------------------------

        x1, y1, x2, y2 = get_guide(
            preview
        )

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        detected, edge_density, variance = (
            detect_id_in_guide(
                preview
            )
        )

        current_time = time.time()

        # ----------------------------------------------------
        # GUIDE COLOR
        # ----------------------------------------------------

        if detected:

            guide_color = (
                0,
                255,
                0
            )

        else:

            guide_color = (
                255,
                180,
                0
            )

        # ----------------------------------------------------
        # DRAW GUIDE
        # ----------------------------------------------------

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            guide_color,
            3
        )

        # ----------------------------------------------------
        # CORNERS
        # ----------------------------------------------------

        corner_length = 35

        thickness = 6

        # Top left
        cv2.line(
            preview,
            (x1, y1),
            (x1 + corner_length, y1),
            guide_color,
            thickness
        )

        cv2.line(
            preview,
            (x1, y1),
            (x1, y1 + corner_length),
            guide_color,
            thickness
        )

        # Top right
        cv2.line(
            preview,
            (x2, y1),
            (x2 - corner_length, y1),
            guide_color,
            thickness
        )

        cv2.line(
            preview,
            (x2, y1),
            (x2, y1 + corner_length),
            guide_color,
            thickness
        )

        # Bottom left
        cv2.line(
            preview,
            (x1, y2),
            (x1 + corner_length, y2),
            guide_color,
            thickness
        )

        cv2.line(
            preview,
            (x1, y2),
            (x1, y2 - corner_length),
            guide_color,
            thickness
        )

        # Bottom right
        cv2.line(
            preview,
            (x2, y2),
            (x2 - corner_length, y2),
            guide_color,
            thickness
        )

        cv2.line(
            preview,
            (x2, y2),
            (x2, y2 - corner_length),
            guide_color,
            thickness
        )

        # ----------------------------------------------------
        # CAMERA TEXT
        # ----------------------------------------------------

        cv2.putText(
            preview,
            "PLACE EMIRATES ID HERE",
            (x1 + 20, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            guide_color,
            2
        )

        # ----------------------------------------------------
        # DETECTION STATE
        # ----------------------------------------------------

        if detected:

            if not self.card_detected:

                self.card_detected = True

                self.stable_start = (
                    current_time
                )

                self.stable_frames = 0

            self.stable_frames += 1

            elapsed = (
                current_time
                - self.stable_start
            )

            remaining = max(
                0,
                STABLE_TIME - elapsed
            )

            if not self.processing:

                self.status_label.setText(
                    f"● ID detected — hold steady "
                    f"{remaining:.1f}s"
                )

                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #F59E0B;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)

            # ------------------------------------------------
            # AUTO CAPTURE
            # ------------------------------------------------

            if (
                elapsed >= STABLE_TIME
                and self.stable_frames >= 10
                and not self.processing
                and (
                    current_time
                    - self.last_scan_time
                ) >= SCAN_COOLDOWN
            ):

                self.last_scan_time = (
                    current_time
                )

                self.processing = True

                captured_frame = frame.copy()

                self.process_id(
                    captured_frame
                )

        else:

            self.card_detected = False

            self.stable_start = None

            self.stable_frames = 0

            if not self.processing:

                self.status_label.setText(
                    "● Place Emirates ID inside the guide"
                )

                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #60A5FA;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)

        # ====================================================
        # DISPLAY CAMERA
        # ====================================================

        rgb = cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        bytes_per_line = (
            ch * w
        )

        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(
            image
        )

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(
            pixmap
        )

    # ========================================================
    # PROCESS ID
    # ========================================================

    def process_id(self, frame):

        self.status_label.setText(
            "● ID captured — AI extracting data..."
        )

        self.status_label.setStyleSheet("""
            QLabel {
                color: #F59E0B;
                font-size: 16px;
                font-weight: bold;
            }
        """)

        self.output_area.setText(
            """
╔══════════════════════════════════════════════╗
║                AI PROCESSING                 ║
╚══════════════════════════════════════════════╝

✓ Emirates ID detected

✓ Image captured automatically

→ Sending image to AI...

→ Extracting information...

Please wait...
"""
        )

        QApplication.processEvents()

        temp_path = None

        try:

            # ------------------------------------------------
            # TEMP IMAGE
            # ------------------------------------------------

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )

            temp_path = temp_file.name

            temp_file.close()

            cv2.imwrite(
                temp_path,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            # ------------------------------------------------
            # OLLAMA
            # ------------------------------------------------

            response = ollama.chat(

                model=OLLAMA_MODEL,

                messages=[

                    {
                        "role": "user",

                        "content": (

                            "You are an Emirates ID OCR "
                            "information extraction system.\n\n"

                            "Analyze the Emirates ID image "
                            "carefully.\n\n"

                            "Extract only:\n\n"

                            "FullName\n"
                            "IDNumber\n"
                            "Nationality\n"
                            "DateOfBirth\n"
                            "IssuingDate\n"
                            "ExpiryDate\n\n"

                            "Return ONLY valid JSON.\n"

                            "No Markdown.\n"
                            "No code blocks.\n"
                            "No explanations.\n\n"

                            "{\n"
                            '  "FullName": "",\n'
                            '  "IDNumber": "",\n'
                            '  "Nationality": "",\n'
                            '  "DateOfBirth": "",\n'
                            '  "IssuingDate": "",\n'
                            '  "ExpiryDate": ""\n'
                            "}"
                        ),

                        "images": [
                            temp_path
                        ]
                    }
                ]
            )

            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            raw_output = response[
                "message"
            ][
                "content"
            ]

            print(
                "\n========== OLLAMA RESPONSE =========="
            )

            print(
                raw_output
            )

            print(
                "======================================\n"
            )

            # ------------------------------------------------
            # PARSE
            # ------------------------------------------------

            parsed = parse_json_response(
                raw_output
            )

            if parsed:

                self.last_parsed = parsed

                self.output_area.setText(
                    format_professional(
                        parsed
                    )
                )

                self.save_btn.setEnabled(
                    True
                )

                self.status_label.setText(
                    "● Extraction complete — ready for next ID"
                )

                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #10B981;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)

            else:

                self.last_parsed = None

                self.output_area.setText(
                    """
╔══════════════════════════════════════════════╗
║              EXTRACTION FAILED               ║
╚══════════════════════════════════════════════╝

The ID was captured but the AI response
could not be converted into structured data.

Please place the ID clearly inside the guide
and try again.
"""
                )

                self.status_label.setText(
                    "● Extraction failed"
                )

                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #EF4444;
                        font-size: 16px;
                        font-weight: bold;
                    }
                """)

        except Exception as e:

            self.last_parsed = None

            self.output_area.setText(
                f"""
╔══════════════════════════════════════════════╗
║                    ERROR                     ║
╚══════════════════════════════════════════════╝

{str(e)}

Check:

• Ollama is running
• Model is available
• Model supports image input
• Camera is working
"""
            )

            self.status_label.setText(
                "● AI processing error"
            )

            self.status_label.setStyleSheet("""
                QLabel {
                    color: #EF4444;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

        finally:

            if (
                temp_path is not None
                and os.path.exists(
                    temp_path
                )
            ):

                try:

                    os.unlink(
                        temp_path
                    )

                except Exception:
                    pass

            self.processing = False

            QTimer.singleShot(
                int(
                    SCAN_COOLDOWN * 1000
                ),
                self.reset_scanner
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset_scanner(self):

        self.card_detected = False

        self.stable_start = None

        self.stable_frames = 0

        if not self.processing:

            self.status_label.setText(
                "● Ready — place next Emirates ID inside guide"
            )

            self.status_label.setStyleSheet("""
                QLabel {
                    color: #60A5FA;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

    # ========================================================
    # SAVE
    # ========================================================

    def save_to_file(self):

        if not self.last_parsed:

            self.output_area.append(
                "\n\nNo structured data available."
            )

            return

        file_path, _ = QFileDialog.getSaveFileName(

            self,

            "Save Emirates ID Data",

            "emirates_id_data.json",

            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.last_parsed,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            self.output_area.append(
                "\n\n✓ Data saved successfully."
            )

            self.output_area.append(
                f"\nFile: {file_path}"
            )

        except Exception as e:

            self.output_area.append(
                f"\n\nSave error:\n{str(e)}"
            )

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.timer.stop()

        if self.cap.isOpened():

            self.cap.release()

        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    app.setStyleSheet("""
        QWidget {
            background-color: #020617;
            color: #F8FAFC;
        }
    """)

    window = IDCaptureApp()

    window.show()

    sys.exit(
        app.exec_()
    )