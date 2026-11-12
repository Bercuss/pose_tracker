# -*- coding: utf-8 -*-
import cv2
import time
import math
import sqlite3
import numpy as np
import pyrealsense2 as rs

from collections import deque
from ultralytics import YOLO


# ----------------------------------------------------------------------
# Beallitasok
# ----------------------------------------------------------------------

MODEL_NAME = "yolov8n-pose.pt"

MIRROR_VIEW = True
MIN_CONFIDENCE = 0.6

MIN_DT = 0.001
MAX_DT = 0.5

ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0

VEL_WINDOW = 0.12
ACC_WINDOW = 0.20

ANGLE_VEL_FLOOR = 6.0
ACC_FLOOR = 150.0

ANGLE_SMOOTH = 0.4
ACC_SMOOTH = 0.4

NORMALIZE_BY_BODY = False

# ----------------------------------------------------------------------
# Intel RealSense kamera beallitasok
# ----------------------------------------------------------------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def side(left_idx, right_idx):
    return (right_idx, left_idx) if MIRROR_VIEW else (left_idx, right_idx)


ARMS = {
    "BAL kar": side((5, 7, 9), (6, 8, 10))[0],
    "JOBB kar": side((5, 7, 9), (6, 8, 10))[1],
}

LEGS = {
    "BAL lab": side((11, 13, 15), (12, 14, 16))[0],
    "JOBB lab": side((11, 13, 15), (12, 14, 16))[1],
}

JOINTS = {
    "Bal csuklo": side(9, 10)[0],
    "Jobb csuklo": side(9, 10)[1],
    "Bal konyok": side(7, 8)[0],
    "Jobb konyok": side(7, 8)[1],
    "Bal terd": side(13, 14)[0],
    "Jobb terd": side(13, 14)[1],
    "Bal boka": side(15, 16)[0],
    "Jobb boka": side(15, 16)[1],
    "Fej": 0,
    "Bal vall": side(5, 6)[0],
    "Jobb vall": side(5, 6)[1],
    "Bal csipo": side(11, 12)[0],
    "Jobb csipo": side(11, 12)[1],
    "Nyak": None,
}

BASE_KEYPOINTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
USED_KEYPOINTS = sorted(
    {i for t in ARMS.values() for i in t}
    | {i for t in LEGS.values() for i in t}
    | {JOINTS[k] for k in JOINTS if JOINTS[k] is not None}
    | BASE_KEYPOINTS
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------
# Adatbazis inicializalasa (bovitve torzs, medence es csipo adatokkal)
# ----------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect("mozgas_adatok.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bal_kar_szog REAL,
            jobb_kar_szog REAL,
            bal_terd_szog REAL,
            jobb_terd_szog REAL,
            bal_csuklo_gyorsulas REAL,
            jobb_csuklo_gyorsulas REAL,
            bal_boka_gyorsulas REAL,
            jobb_boka_gyorsulas REAL,
            fej_gyorsulas REAL,
            nyak_gyorsulas REAL,
            bal_vall_gyorsulas REAL,
            jobb_vall_gyorsulas REAL,
            bal_csipo_gyorsulas REAL,
            jobb_csipo_gyorsulas REAL,
            vall_szog REAL,
            medence_szog REAL,
            torzs_dolesszog REAL
        )
    """)
    
    existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(measurements)").fetchall()]
    new_cols = [
        ("fej_gyorsulas", "REAL"),
        ("nyak_gyorsulas", "REAL"),
        ("bal_vall_gyorsulas", "REAL"),
        ("jobb_vall_gyorsulas", "REAL"),
        ("bal_csipo_gyorsulas", "REAL"),
        ("jobb_csipo_gyorsulas", "REAL"),
        ("vall_szog", "REAL"),
        ("medence_szog", "REAL"),
        ("torzs_dolesszog", "REAL")
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE measurements ADD COLUMN {col_name} {col_type}")

    conn.commit()
    return conn


# ----------------------------------------------------------------------
# Szogszamitas
# ----------------------------------------------------------------------

def get_angle(a, b, c):
    rad = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    deg = abs(rad * 180.0 / np.pi)
    return 360.0 - deg if deg > 180.0 else deg


# ----------------------------------------------------------------------
# One Euro Filter
# ----------------------------------------------------------------------

class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)

        if self.t_prev is None or (t - self.t_prev) > MAX_DT:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dt = max(t - self.t_prev, MIN_DT)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


# ----------------------------------------------------------------------
# Idoablakos derivalt
# ----------------------------------------------------------------------

class WindowBuffer:
    def __init__(self, window):
        self.window = window
        self.buf = deque()

    def push(self, t, v):
        self.buf.append((t, v))
        while len(self.buf) > 2 and t - self.buf[1][0] >= self.window:
            self.buf.popleft()

    def derivative(self):
        t1, v1 = self.buf[-1]
        t0, v0 = self.buf[0]
        dt = t1 - t0

        if dt < self.window * 0.5:
            return None

        return (v1 - v0) / dt


# ----------------------------------------------------------------------
# Szogsebesseg koveto
# ----------------------------------------------------------------------

class AngleTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buf = WindowBuffer(VEL_WINDOW)
        self.last_time = None
        self.velocity = 0.0
        self.has_value = False

    def update(self, angle, now):
        if self.last_time is not None and (now - self.last_time) > MAX_DT:
            self.reset()

        self.last_time = now
        self.buf.push(now, angle)
        d = self.buf.derivative()

        if d is None:
            return

        if abs(d) < ANGLE_VEL_FLOOR:
            d = 0.0

        if self.has_value:
            self.velocity = (1 - ANGLE_SMOOTH) * self.velocity + ANGLE_SMOOTH * d
        else:
            self.velocity = d
            self.has_value = True


# ----------------------------------------------------------------------
# Izuleti sebesseg es gyorsulas koveto
# ----------------------------------------------------------------------

class PointTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos_buf = WindowBuffer(VEL_WINDOW)
        self.vel_buf = WindowBuffer(ACC_WINDOW)
        self.last_time = None
        self.speed = None
        self.acceleration = None

    def update(self, pos, now, acc_floor):
        if self.last_time is not None and (now - self.last_time) > MAX_DT:
            self.reset()

        self.last_time = now
        self.pos_buf.push(now, np.asarray(pos, dtype=float))
        vel = self.pos_buf.derivative()

        if vel is None:
            return

        self.speed = float(np.hypot(vel[0], vel[1]))
        self.vel_buf.push(now, vel)
        acc = self.vel_buf.derivative()

        if acc is None:
            return

        value = float(np.hypot(acc[0], acc[1]))

        if value < acc_floor:
            value = 0.0

        if self.acceleration is None:
            self.acceleration = value
        else:
            self.acceleration = (1 - ACC_SMOOTH) * self.acceleration + ACC_SMOOTH * value


# ----------------------------------------------------------------------
# Szemely kivalasztasa
# ----------------------------------------------------------------------

def pick_person(result):
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    idx = 0
    body_h = None

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(areas.argmax())
        body_h = float(boxes[idx, 3] - boxes[idx, 1])

    points = result.keypoints.xy[idx].cpu().numpy()
    confs = result.keypoints.conf[idx].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(points))

    return points, confs, body_h


# ----------------------------------------------------------------------
# Informacios panel kirajzolasa
# ----------------------------------------------------------------------

def draw_panel(frame, lines):
    height = 14 + 22 * len(lines)
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (440, 10 + height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (20, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += 22


# ----------------------------------------------------------------------
# Kamera inicializalasa
# ----------------------------------------------------------------------

def start_camera():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.bgr8,
            CAMERA_FPS
        )
        print("Intel RealSense kamera keresese...")
        pipeline.start(config)
        print("RealSense kamera sikeresen elinditva!")
        return pipeline, "realsense"
    except Exception as e:
        print(f"RealSense nem talalhato ({e}), atvaltas sima webkamerara...")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Sima webkamera sikeresen elinditva!")
        return cap, "webcam"
    
    raise Exception("Nem talalhato sem Intel RealSense, sem sima webkamera!")


# ----------------------------------------------------------------------
# Foprogram
# ----------------------------------------------------------------------

def main():
    print("YOLO modell betoltese...")
    model = YOLO(MODEL_NAME)

    db_conn = init_db()
    db_cursor = db_conn.cursor()

    pipeline = None
    camera_type = None
    try:
        pipeline, camera_type = start_camera()
    except Exception as e:
        print(f"Kamerainditasi hiba: {e}")
        db_conn.close()
        return

    kp_filters = {
        i: OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        for i in USED_KEYPOINTS
    }

    angle_trackers = {
        name: AngleTracker()
        for name in {**ARMS, **LEGS}
    }

    point_trackers = {
        name: PointTracker()
        for name in JOINTS if name != "Nyak"
    }
    point_trackers["Nyak"] = PointTracker()

    body_h_ema = None

    def reset_all():
        nonlocal body_h_ema
        for f in kp_filters.values():
            f.reset()
        for t in angle_trackers.values():
            t.reset()
        for t in point_trackers.values():
            t.reset()
        body_h_ema = None

    GREEN = (0, 255, 0)
    ORANGE = (255, 200, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    GRAY = (160, 160, 160)

    unit = "testmag/s^2" if NORMALIZE_BY_BODY else "px/s^2"

    try:
        while True:
            if camera_type == "realsense":
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = pipeline.read()
                if not ret or frame is None:
                    print("Hiba a webkamera olvasasakor!")
                    break
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()

            result = model(frame, verbose=False)[0]
            annotated = result.plot()
            person = pick_person(result)

            current_db_data = {
                "bal_kar_szog": None,
                "jobb_kar_szog": None,
                "bal_terd_szog": None,
                "jobb_terd_szog": None,
                "bal_csuklo_gyorsulas": None,
                "jobb_csuklo_gyorsulas": None,
                "bal_boka_gyorsulas": None,
                "jobb_boka_gyorsulas": None,
                "fej_gyorsulas": None,
                "nyak_gyorsulas": None,
                "bal_vall_gyorsulas": None,
                "jobb_vall_gyorsulas": None,
                "bal_csipo_gyorsulas": None,
                "jobb_csipo_gyorsulas": None,
                "vall_szog": None,
                "medence_szog": None,
                "torzs_dolesszog": None
            }

            if person is None:
                reset_all()
                cv2.putText(annotated, "Nem lathato ember", (20, 35), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                points, confs, body_h = person
                lines = []

                fp = {}
                for i in USED_KEYPOINTS:
                    if i < len(confs) and confs[i] >= MIN_CONFIDENCE:
                        fp[i] = kp_filters[i](points[i], now)
                    else:
                        if i in kp_filters:
                            kp_filters[i].reset()
                        fp[i] = None

                scale = 1.0
                if NORMALIZE_BY_BODY and body_h:
                    body_h_ema = body_h if body_h_ema is None else 0.9 * body_h_ema + 0.1 * body_h
                    scale = body_h_ema

                acc_floor = ACC_FLOOR / scale

                lines.append(("--- Vegtagok (szog) ---", GRAY))
                limb_colors = {"BAL kar": GREEN, "JOBB kar": ORANGE, "BAL lab": GREEN, "JOBB lab": ORANGE}

                all_limbs = {**ARMS, **LEGS}
                for name, (s, e, w) in all_limbs.items():
                    tracker = angle_trackers[name]
                    if s in fp and e in fp and w in fp and fp[s] is not None and fp[e] is not None and fp[w] is not None:
                        angle = get_angle(fp[s], fp[e], fp[w])
                        tracker.update(angle, now)

                        if name == "BAL kar": current_db_data["bal_kar_szog"] = angle
                        elif name == "JOBB kar": current_db_data["jobb_kar_szog"] = angle
                        elif name == "BAL lab": current_db_data["bal_terd_szog"] = angle
                        elif name == "JOBB lab": current_db_data["jobb_terd_szog"] = angle

                        cv2.putText(annotated, f"{int(angle)} deg", (int(fp[e][0]), int(fp[e][1])), FONT, 0.5, YELLOW, 2, cv2.LINE_AA)
                        lines.append((f"{name}: {int(angle)} deg {int(tracker.velocity):+d} deg/s", limb_colors[name]))
                    else:
                        tracker.reset()
                        lines.append((f"{name}: nem lathato", GRAY))

                # Vallak es Nyak
                left_shoulder_raw_idx = side(5, 6)[0]
                right_shoulder_raw_idx = side(5, 6)[1]
                neck_pos = None

                if left_shoulder_raw_idx in fp and right_shoulder_raw_idx in fp and fp[left_shoulder_raw_idx] is not None and fp[right_shoulder_raw_idx] is not None:
                    ls = fp[left_shoulder_raw_idx]
                    rs_pt = fp[right_shoulder_raw_idx]
                    dx = rs_pt[0] - ls[0]
                    dy = rs_pt[1] - ls[1]
                    shoulder_angle = math.degrees(math.atan2(dy, dx))
                    current_db_data["vall_szog"] = shoulder_angle
                    lines.append((f"Vall dolesszog: {shoulder_angle:.1f} deg", WHITE))
                    
                    neck_pos = ((ls[0] + rs_pt[0]) / 2.0, (ls[1] + rs_pt[1]) / 2.0)
                    point_trackers["Nyak"].update((neck_pos[0] / scale, neck_pos[1] / scale), now, acc_floor)
                    if point_trackers["Nyak"].acceleration is not None:
                        current_db_data["nyak_gyorsulas"] = point_trackers["Nyak"].acceleration

                # Medence es Torzs
                left_hip_idx = side(11, 12)[0]
                right_hip_idx = side(11, 12)[1]

                if left_hip_idx in fp and right_hip_idx in fp and fp[left_hip_idx] is not None and fp[right_hip_idx] is not None:
                    lh = fp[left_hip_idx]
                    rh = fp[right_hip_idx]
                    dx_hip = rh[0] - lh[0]
                    dy_hip = rh[1] - lh[1]
                    pelvis_angle = math.degrees(math.atan2(dy_hip, dx_hip))
                    current_db_data["medence_szog"] = pelvis_angle
                    lines.append((f"Medence dolesszog: {pelvis_angle:.1f} deg", WHITE))

                    hip_center = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
                    if neck_pos is not None:
                        dx_torso = hip_center[0] - neck_pos[0]
                        dy_torso = hip_center[1] - neck_pos[1]
                        torso_angle = math.degrees(math.atan2(abs(dx_torso), abs(dy_torso)))
                        current_db_data["torzs_dolesszog"] = torso_angle
                        lines.append((f"Torzs doles: {torso_angle:.1f} deg", WHITE))

                lines.append(("--- Fej, Nyak, Torzs (gyorsulas) ---", GRAY))

                head_idx = 0
                if head_idx in fp and fp[head_idx] is not None:
                    hp = fp[head_idx]
                    cv2.circle(annotated, (int(hp[0]), int(hp[1])), 5, YELLOW, -1)
                    point_trackers["Fej"].update((hp[0] / scale, hp[1] / scale), now, acc_floor)
                    if point_trackers["Fej"].acceleration is not None:
                        current_db_data["fej_gyorsulas"] = point_trackers["Fej"].acceleration
                        lines.append((f"Fej gyorsulas: {point_trackers['Fej'].acceleration:.0f} {unit}", GREEN))

                if point_trackers["Nyak"].acceleration is not None:
                    lines.append((f"Nyak gyorsulas: {point_trackers['Nyak'].acceleration:.0f} {unit}", GREEN))

                for name, idx in JOINTS.items():
                    if idx is None or name == "Fej":
                        continue
                    tracker = point_trackers[name]
                    p = fp[idx] if idx in fp else None

                    if p is None:
                        tracker.reset()
                        continue

                    cv2.circle(annotated, (int(p[0]), int(p[1])), 5, YELLOW, -1)
                    tracker.update((p[0] / scale, p[1] / scale), now, acc_floor)

                    if tracker.acceleration is not None:
                        v = tracker.acceleration
                        if name == "Bal csuklo": current_db_data["bal_csuklo_gyorsulas"] = v
                        elif name == "Jobb csuklo": current_db_data["jobb_csuklo_gyorsulas"] = v
                        elif name == "Bal boka": current_db_data["bal_boka_gyorsulas"] = v
                        elif name == "Jobb boka": current_db_data["jobb_boka_gyorsulas"] = v
                        elif name == "Bal vall": current_db_data["bal_vall_gyorsulas"] = v
                        elif name == "Jobb vall": current_db_data["jobb_vall_gyorsulas"] = v
                        elif name == "Bal csipo": current_db_data["bal_csipo_gyorsulas"] = v
                        elif name == "Jobb csipo": current_db_data["jobb_csipo_gyorsulas"] = v

                        lines.append((f"{name}: {v:.0f} {unit}", GREEN))

                draw_panel(annotated, lines)

                db_cursor.execute(
                    """
                    INSERT INTO measurements (
                        bal_kar_szog,
                        jobb_kar_szog,
                        bal_terd_szog,
                        jobb_terd_szog,
                        bal_csuklo_gyorsulas,
                        jobb_csuklo_gyorsulas,
                        bal_boka_gyorsulas,
                        jobb_boka_gyorsulas,
                        fej_gyorsulas,
                        nyak_gyorsulas,
                        bal_vall_gyorsulas,
                        jobb_vall_gyorsulas,
                        bal_csipo_gyorsulas,
                        jobb_csipo_gyorsulas,
                        vall_szog,
                        medence_szog,
                        torzs_dolesszog
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current_db_data["bal_kar_szog"],
                        current_db_data["jobb_kar_szog"],
                        current_db_data["bal_terd_szog"],
                        current_db_data["jobb_terd_szog"],
                        current_db_data["bal_csuklo_gyorsulas"],
                        current_db_data["jobb_csuklo_gyorsulas"],
                        current_db_data["bal_boka_gyorsulas"],
                        current_db_data["jobb_boka_gyorsulas"],
                        current_db_data["fej_gyorsulas"],
                        current_db_data["nyak_gyorsulas"],
                        current_db_data["bal_vall_gyorsulas"],
                        current_db_data["jobb_vall_gyorsulas"],
                        current_db_data["bal_csipo_gyorsulas"],
                        current_db_data["jobb_csipo_gyorsulas"],
                        current_db_data["vall_szog"],
                        current_db_data["medence_szog"],
                        current_db_data["torzs_dolesszog"]
                    )
                )
                db_conn.commit()

            cv2.imshow("AI Mozgaskoveto - Intel RealSense", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_all()

    except KeyboardInterrupt:
        print("Program megszakitva.")
    except Exception as e:
        print(f"Feldolgozasi hiba: {e}")
    finally:
        if pipeline is not None:
            if camera_type == "realsense":
                pipeline.stop()
            else:
                pipeline.release()
        db_conn.close()
        cv2.destroyAllWindows()
        print("Kamera leallitva.")
        print("Adatbazis kapcsolat lezarva.")


if __name__ == "__main__":
    main()
