from collections import defaultdict, deque

from pathlib import Path
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import pandas as pd
import sqlite3

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
# model = YOLO("runs/detect/train2/weights/best.pt")

BASE_DIR = Path(__file__).resolve().parent
model = YOLO(str(BASE_DIR / "models" / "best.pt"))

classes = model.names

# Per-class confidence thresholds (gloves: small object, often lower conf)
CONF_GLOVES = 0.25
CONF_HELMET = 0.35
CONF_VEST = 0.35
CONF_NEGATIVE = 0.35  # no_helmet, no_gloves
CONF_PERSON = 0.5
IMGSZ = 832  # Higher resolution helps small objects like gloves (640→832 or 1024)
CONF_INFERENCE = (
    0.15  # Low global conf so gloves (often low-conf) pass through; filtering below
)

# Temporal smoothing for gloves
GLOVES_HISTORY_LEN = 10
GLOVES_ON_THRESHOLD = 3  # 3 detections in last N to turn ON
GLOVES_RECENT_MIN = 1  # At least 1 in last 4 frames
GLOVES_STICKY_COUNT = 6  # Stay ON 6 frames once confirmed


def passes_class_conf(cls, conf):
    """Apply per-class confidence threshold."""
    if cls == "gloves":
        return conf >= CONF_GLOVES
    if cls == "helmet":
        return conf >= CONF_HELMET
    if cls == "vest":
        return conf >= CONF_VEST
    if cls in ("no_helmet", "no_gloves"):
        return conf >= CONF_NEGATIVE
    if cls == "Person":
        return conf >= CONF_PERSON
    return False


# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect("safety.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS safety_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT,
            helmet INTEGER,
            gloves INTEGER,
            vest INTEGER,
            status TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ---------------- GLOBAL ANALYSIS DATA ----------------
analysis_data = []

# Per-person gloves history for temporal smoothing (avoids true/false flicker)
person_gloves_history = defaultdict(lambda: deque(maxlen=GLOVES_HISTORY_LEN))
person_gloves_sticky = {}  # track_id -> frames left to stay True


def is_inside(item_box, person_box):
    """Strict: PPE box top-left must be inside person box."""
    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = item_box[:4]
    return px1 < x1 < px2 and py1 < y1 < py2


def gloves_near_person(glove_box, person_box, strict=False):
    """
    Check if glove detection is associated with person.
    strict=False (positive gloves): relaxed - hands at sides/waist/raised all count.
    strict=True (no_gloves): stricter - avoid false override.
    """
    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = glove_box[:4]
    gcx = (x1 + x2) / 2
    gcy = (y1 + y2) / 2
    w, h = px2 - px1, py2 - py1
    if strict:
        # no_gloves: stricter to avoid wrong override
        margin = 0.2
        min_overlap = 0.35
    else:
        # positive gloves: relaxed - hands can extend (arms at sides, waist, etc.)
        margin = 0.5
        min_overlap = 0.15
    ex1 = px1 - w * margin
    ex2 = px2 + w * margin
    ey1 = py1 - h * margin
    ey2 = py2 + h * margin
    if not (ex1 <= gcx <= ex2 and ey1 <= gcy <= ey2):
        return False
    ix1 = max(px1, x1)
    iy1 = max(py1, y1)
    ix2 = min(px2, x2)
    iy2 = min(py2, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    glove_area = (x2 - x1) * (y2 - y1)
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return inter_area >= min_overlap * glove_area


def generate_frames():
    global analysis_data
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Low global conf to get more raw predictions; per-class filtering below
        results = model.track(frame, conf=CONF_INFERENCE, imgsz=IMGSZ, persist=True)[0]

        persons = []
        ppe_positive = {"helmet": [], "gloves": [], "vest": []}
        ppe_negative = {"no_helmet": [], "no_gloves": []}

        for box in results.boxes:
            cls = classes[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if not passes_class_conf(cls, conf):
                continue
            if cls == "Person":
                persons.append((x1, y1, x2, y2))
            elif cls in ppe_positive:
                ppe_positive[cls].append((x1, y1, x2, y2, conf))
            elif cls in ppe_negative:
                ppe_negative[cls].append((x1, y1, x2, y2, conf))

        analysis_data = []

        # -------- OPEN DB CONNECTION (PER FRAME) --------
        conn = sqlite3.connect("safety.db")
        c = conn.cursor()

        analysis_data = []

        conn = sqlite3.connect("safety.db")
        c = conn.cursor()

        for box in results.boxes:
            cls = classes[int(box.cls[0])]
            if cls != "Person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            worker_id = f"Person_{track_id}"
            status = {"helmet": False, "gloves": False, "vest": False}
            confs = []

            for item in ppe_positive:
                for pbox in ppe_positive[item]:
                    if item == "gloves":
                        match = gloves_near_person(pbox, (x1, y1, x2, y2), strict=False)
                    else:
                        match = is_inside(pbox, (x1, y1, x2, y2))
                    if match:
                        status[item] = True
                        confs.append(pbox[4])
            for neg in ppe_negative:
                for nbox in ppe_negative[neg]:
                    if neg == "no_gloves":
                        match = gloves_near_person(nbox, (x1, y1, x2, y2), strict=True)
                    else:
                        match = is_inside(nbox, (x1, y1, x2, y2))
                    if match:
                        if neg == "no_helmet":
                            status["helmet"] = False
                        elif neg == "no_gloves":
                            status["gloves"] = False

            # Temporal smoothing: reduce false positives, stay stable when gloves ON
            raw_gloves = status["gloves"]
            no_gloves_override = any(
                gloves_near_person(nbox, (x1, y1, x2, y2), strict=True)
                for nbox in ppe_negative["no_gloves"]
            )
            hist = person_gloves_history[track_id]
            hist.append(1 if raw_gloves else 0)
            sticky = person_gloves_sticky.get(track_id, 0)
            recent = list(hist)[-4:]  # last 4 frames
            total_hist = sum(hist)
            recent_count = sum(recent)
            if no_gloves_override:
                person_gloves_sticky[track_id] = 0
                status["gloves"] = False
            elif raw_gloves:
                person_gloves_sticky[track_id] = GLOVES_STICKY_COUNT
                status["gloves"] = True
            elif sticky > 0:
                person_gloves_sticky[track_id] = sticky - 1
                status["gloves"] = True
            elif (
                total_hist >= GLOVES_ON_THRESHOLD and recent_count >= GLOVES_RECENT_MIN
            ):
                status["gloves"] = True
                person_gloves_sticky[track_id] = GLOVES_STICKY_COUNT

            avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
            safe = all(status.values())

            analysis_data.append(
                {
                    "Person": worker_id,
                    "Helmet": status["helmet"],
                    "Gloves": status["gloves"],
                    "Vest": status["vest"],
                    "Status": "Safe" if safe else "Unsafe",
                    "Confidence": avg_conf,
                }
            )

            c.execute(
                """
                INSERT INTO safety_logs
                (worker_id, helmet, gloves, vest, status, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    int(status["helmet"]),
                    int(status["gloves"]),
                    int(status["vest"]),
                    "Safe" if safe else "Unsafe",
                    avg_conf,
                ),
            )  # Commit the database transaction
            color = (0, 255, 0) if safe else (0, 0, 255)
            label = f"{worker_id} | {'SAFE' if safe else 'UNSAFE'} ({avg_conf})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        conn.commit()
        conn.close()

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/demo")
def demo():
    return render_template("monitor.html")


@app.route("/dashboard_data")
def dashboard_data():
    df = pd.DataFrame(analysis_data)
    stats = {
        "total": len(df),
        "safe": (df["Status"] == "Safe").sum() if not df.empty else 0,
        "unsafe": (df["Status"] == "Unsafe").sum() if not df.empty else 0,
    }
    return render_template(
        "dashboard_table.html",
        table=df.to_dict(orient="records"),
        stats=stats,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/technology")
def technology():
    return render_template("technology.html")


if __name__ == "__main__":
    app.run(debug=True)
