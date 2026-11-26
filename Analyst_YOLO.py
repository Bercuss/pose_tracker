import cv2
import time
import math
import json
import sqlite3
from pathlib import Path
import numpy as np
from collections import deque
from ultralytics import YOLO


# Beállítások
MODEL_NAME = "yolov8n-pose.pt"
DATASET_DIR = Path("squat_dataset")
KNN_K = 3                            

# ÚJ: Guggolás detektálási küszöbök (egyeznek a record_squats.py-vel)
START_KNEE_ANGLE = 155.0
END_KNEE_ANGLE = 162.0
MIN_VALID_DEPTH = 140.0
MIN_FRAMES = 10

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

def side(left_idx, right_idx):
    return (right_idx, left_idx) if MIRROR_VIEW else (left_idx, right_idx)

ARMS = {
    "BAL kar": side((5, 7, 9), (6, 8, 10))[0],
    "JOBB kar": side((5, 7, 9), (6, 8, 10))[1],
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
}

SQUAT_SIDES = {
    "left":  (5, 11, 13, 15),
    "right": (6, 12, 14, 16),
}

USED_KEYPOINTS = sorted(
    {i for t in ARMS.values() for i in t}
    | set(JOINTS.values())
    | {i for t in SQUAT_SIDES.values() for i in t}
)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Adatbázis
def init_db():
    conn = sqlite3.connect("mozgas_adatok.db")
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS measurements')
    cursor.execute('''
        CREATE TABLE measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bal_kar_szog REAL,
            jobb_kar_szog REAL,
            bal_csuklo_gyorsulas REAL,
            jobb_csuklo_gyorsulas REAL
        )
    ''')
    conn.commit()
    return conn

# Szűrők, geometriai függvények és trackerek
def get_angle(a, b, c):
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / np.pi)
    return 360.0 - deg if deg > 180.0 else deg

# Törzs előredőlése a függőlegeshez képest (fokban)
def get_torso_lean(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    return abs(math.degrees(math.atan2(dx, dy)))

# Jobban látható oldal kiválasztása guggoláshoz
def select_best_side(fp, confs):
    best = None
    best_conf = -1.0
    for side_name, (sh, hip, knee, ankle) in SQUAT_SIDES.items():
        if all(fp.get(i) is not None for i in (sh, hip, knee, ankle)):
            avg_c = float(np.mean([confs[sh], confs[hip], confs[knee], confs[ankle]]))
            if avg_c > best_conf:
                best_conf = avg_c
                best = (side_name, sh, hip, knee, ankle)
    return best

# ÚJ: Referencia JSON-ok betöltése és DTW + k-NN kiértékelés
def load_squat_dataset(dataset_dir: Path):
    samples = []
    if not dataset_dir.exists():
        return samples

    for filepath in sorted(dataset_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = data.get("label")
            frames = data.get("frames", [])
            if label not in ("Y", "N") or len(frames) < 2:
                continue
            seq = np.array(
                [[fr["knee_angle"], fr["hip_angle"], fr["torso_lean"]] for fr in frames],
                dtype=float
            )
            samples.append({
                "code": data.get("code", filepath.stem),
                "label": label,
                "note": data.get("note", ""),
                "seq": seq,
            })
        except Exception as e:
            print(f"Figyelmeztetes: {filepath.name} betoltese sikertelen ({e})")

    return samples

def dtw_distance(seq_a, seq_b):
    """Dynamic Time Warping: visszaadja az átlagos szögeltérést (fok) és hasonlóságot (%)."""
    A = np.asarray(seq_a, dtype=float)
    B = np.asarray(seq_b, dtype=float)
    n, m = len(A), len(B)

    # Képkockapárok átlagos szögeltérése [térd, csípő, törzsdőlés]
    C = np.mean(np.abs(A[:, None, :] - B[None, :, :]), axis=2)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Optimális útvonal hosszának visszakövetése a normalizáláshoz
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        prev = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        best = int(np.argmin(prev))
        if best == 0:
            i -= 1; j -= 1
        elif best == 1:
            i -= 1
        else:
            j -= 1

    mean_error = float(D[n, m] / max(steps, 1))
    similarity = 100.0 * math.exp(-mean_error / 25.0)
    return mean_error, similarity

def classify_squat(live_seq, dataset, k=KNN_K):
    """DTW + távolsággal súlyozott k-NN osztályozás a tárolt Y és N minták alapján."""
    scored = []
    for sample in dataset:
        err, sim = dtw_distance(sample["seq"], live_seq)
        scored.append((err, sim, sample))

    scored.sort(key=lambda x: x[0])
    top_k = scored[:min(k, len(scored))]

    weights = {"Y": 0.0, "N": 0.0}
    for err, sim, sample in top_k:
        weights[sample["label"]] += 1.0 / (err + 1e-6)

    verdict = "Y" if weights["Y"] >= weights["N"] else "N"
    total_w = weights["Y"] + weights["N"]
    confidence = (weights[verdict] / total_w) * 100.0 if total_w > 0 else 0.0
    return verdict, confidence, top_k

class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff; self.reset()
    def reset(self):
        self.t_prev = None; self.x_prev = None; self.dx_prev = None
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.t_prev is None or (t - self.t_prev) > MAX_DT:
            self.t_prev = t; self.x_prev = x; self.dx_prev = np.zeros_like(x)
            return x
        dt = max(t - self.t_prev, MIN_DT)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev = t; self.x_prev = x_hat; self.dx_prev = dx_hat
        return x_hat

class WindowBuffer:
    def __init__(self, window):
        self.window = window; self.buf = deque()
    def push(self, t, v):
        self.buf.append((t, v))
        while len(self.buf) > 2 and t - self.buf[1][0] >= self.window:
            self.buf.popleft()
    def derivative(self):
        t1, v1 = self.buf[-1]; t0, v0 = self.buf[0]
        dt = t1 - t0
        if dt < self.window * 0.5: return None
        return (v1 - v0) / dt

class AngleTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.buf = WindowBuffer(VEL_WINDOW); self.last_time = None
        self.velocity = 0.0; self.has_value = False
    def update(self, angle, now):
        if self.last_time is not None and (now - self.last_time) > MAX_DT: self.reset()
        self.last_time = now; self.buf.push(now, angle)
        d = self.buf.derivative()
        if d is None: return
        if abs(d) < ANGLE_VEL_FLOOR: d = 0.0
        if self.has_value: self.velocity = (1 - ANGLE_SMOOTH) * self.velocity + ANGLE_SMOOTH * d
        else: self.velocity = d; self.has_value = True

class PointTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.pos_buf = WindowBuffer(VEL_WINDOW); self.vel_buf = WindowBuffer(ACC_WINDOW)
        self.last_time = None; self.speed = None; self.acceleration = None
    def update(self, pos, now, acc_floor):
        if self.last_time is not None and (now - self.last_time) > MAX_DT: self.reset()
        self.last_time = now; self.pos_buf.push(now, np.asarray(pos, dtype=float))
        vel = self.pos_buf.derivative()
        if vel is None: return
        self.speed = float(np.hypot(vel[0], vel[1]))
        self.vel_buf.push(now, vel)
        acc = self.vel_buf.derivative()
        if acc is None: return
        value = float(np.hypot(acc[0], acc[1]))
        if value < acc_floor: value = 0.0
        if self.acceleration is None: self.acceleration = value
        else: self.acceleration = (1 - ACC_SMOOTH) * self.acceleration + ACC_SMOOTH * value

def pick_person(result):
    if result.keypoints is None or len(result.keypoints.xy) == 0: return None
    idx = 0; body_h = None
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(areas.argmax())
        body_h = float(boxes[idx, 3] - boxes[idx, 1])
    points = result.keypoints.xy[idx].cpu().numpy()
    if result.keypoints.conf is not None: confs = result.keypoints.conf[idx].cpu().numpy()
    else: confs = np.ones(len(points))
    return points, confs, body_h

def draw_panel(frame, lines):
    height = 14 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (430, 10 + height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 35
    for text, color in lines:
        cv2.putText(frame, text, (20, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 24

# Főprogram
def main():
    #Előre rögzített guggolások betöltése
    dataset = load_squat_dataset(DATASET_DIR)
    if not dataset:
        print(f"HIBA: Nem talalhato Y/N JSON minta a '{DATASET_DIR.resolve()}' mappaban!")
        print("Elobb futtasd a record_squats.py programot nehany minta rogzitesehez.")
        return

    y_cnt = sum(1 for s in dataset if s["label"] == "Y")
    n_cnt = sum(1 for s in dataset if s["label"] == "N")
    print(f"Betoltott referencia mintak: {len(dataset)} db (Helyes/Y: {y_cnt}, Helytelen/N: {n_cnt})")
    if y_cnt == 0 or n_cnt == 0:
        print("Figyelem: Mindket osztalybol (Y es N) erdemes mintat rogziteni a megbizhato donteshez!")

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(0)

    db_conn = init_db()
    db_cursor = db_conn.cursor()

    if not cap.isOpened():
        print("Nem sikerult megnyitni a kamerat.")
        db_conn.close()
        return

    kp_filters = {i: OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF) for i in USED_KEYPOINTS}
    angle_trackers = {name: AngleTracker() for name in ARMS}
    point_trackers = {name: PointTracker() for name in JOINTS}
    body_h_ema = None

    #Aktuális guggolás ideiglenes állapota (csak memóriában, nem mentődik)
    squat_state = "IDLE"
    current_squat_seq = []
    min_knee_seen = 180.0

    def reset_all():
        nonlocal body_h_ema, squat_state, current_squat_seq, min_knee_seen
        for f in kp_filters.values(): f.reset()
        for t in angle_trackers.values(): t.reset()
        for t in point_trackers.values(): t.reset()
        body_h_ema = None
        squat_state = "IDLE"
        current_squat_seq = []
        min_knee_seen = 180.0

    GREEN, RED, ORANGE, YELLOW, WHITE, GRAY = (0, 255, 0), (0, 0, 255), (255, 200, 0), (0, 255, 255), (255, 255, 255), (160, 160, 160)
    unit = "testmag/s^2" if NORMALIZE_BY_BODY else "px/s^2"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        result = model(frame, verbose=False)[0]
        annotated = result.plot()
        person = pick_person(result)

        current_db_data = {
            "bal_kar_szog": None,
            "jobb_kar_szog": None,
            "bal_csuklo_gyorsulas": None,
            "jobb_csuklo_gyorsulas": None
        }

        squat_finished = False

        if person is None:
            reset_all()
            cv2.putText(annotated, "Nem lathato ember", (20, 35), FONT, 0.8, RED, 2, cv2.LINE_AA)
        else:
            points, confs, body_h = person
            lines = []

            fp = {}
            for i in USED_KEYPOINTS:
                if confs[i] >= MIN_CONFIDENCE: fp[i] = kp_filters[i](points[i], now)
                else: kp_filters[i].reset(); fp[i] = None

            scale = 1.0
            if NORMALIZE_BY_BODY and body_h:
                body_h_ema = body_h if body_h_ema is None else 0.9 * body_h_ema + 0.1 * body_h
                scale = body_h_ema
            acc_floor = ACC_FLOOR / scale

            lines.append(("--- Karok (szog, szogsebesseg) ---", GRAY))
            arm_colors = {"BAL kar": GREEN, "JOBB kar": ORANGE}

            for name, (s, e, w) in ARMS.items():
                tracker = angle_trackers[name]
                if fp[s] is not None and fp[e] is not None and fp[w] is not None:
                    angle = get_angle(fp[s], fp[e], fp[w])
                    tracker.update(angle, now)

                    if name == "BAL kar": current_db_data["bal_kar_szog"] = angle
                    if name == "JOBB kar": current_db_data["jobb_kar_szog"] = angle

                    cv2.putText(annotated, f"{int(angle)} deg", (int(fp[e][0]), int(fp[e][1])), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
                    lines.append((f"{name}: {int(angle)} deg  {int(tracker.velocity):+d} deg/s", arm_colors[name]))
                else:
                    tracker.reset()
                    lines.append((f"{name}: nem lathato", GRAY))

            lines.append((f"--- Izuletek (gyorsulas, {unit}) ---", GRAY))

            for name, idx in JOINTS.items():
                tracker = point_trackers[name]
                p = fp[idx]
                if p is None:
                    tracker.reset(); lines.append((f"{name}: nem lathato", GRAY))
                    continue

                cv2.circle(annotated, (int(p[0]), int(p[1])), 5, YELLOW, -1)
                tracker.update((p[0] / scale, p[1] / scale), now, acc_floor)

                if tracker.acceleration is not None:
                    v = tracker.acceleration
                    if name == "Bal csuklo": current_db_data["bal_csuklo_gyorsulas"] = v
                    if name == "Jobb csuklo": current_db_data["jobb_csuklo_gyorsulas"] = v

                    text = f"{name}: {v:.2f} {unit}" if NORMALIZE_BY_BODY else f"{name}: {v:.0f} {unit}"
                    lines.append((text, GREEN))
                elif tracker.speed is not None:
                    lines.append((f"{name}: meres...", YELLOW))
                else:
                    lines.append((f"{name}: varakozas...", WHITE))

            # --- ÚJ: Guggolás figyelése (memóriában, mentés nélkül) ---
            lines.append(("--- Guggolas ellenorzes (DTW + k-NN) ---", GRAY))
            side_data = select_best_side(fp, confs)

            if side_data is None:
                lines.append(("Guggolas: allj oldalra (vall-csipo-terd-boka)", YELLOW))
            else:
                side_name, sh, hip, knee, ankle = side_data
                knee_ang = get_angle(fp[hip], fp[knee], fp[ankle])
                hip_ang = get_angle(fp[sh], fp[hip], fp[knee])
                torso_lean = get_torso_lean(fp[sh], fp[hip])

                cv2.putText(annotated, f"{int(knee_ang)} deg", (int(fp[knee][0]), int(fp[knee][1])),
                            FONT, 0.7, YELLOW, 2, cv2.LINE_AA)
                lines.append((f"Terd: {int(knee_ang)} | Csipo: {int(hip_ang)} | Doles: {int(torso_lean)}", WHITE))

                if squat_state == "IDLE":
                    lines.append(("Allapot: Varja a guggolast...", WHITE))
                    if knee_ang < START_KNEE_ANGLE:
                        squat_state = "RECORDING"
                        current_squat_seq = []
                        min_knee_seen = knee_ang

                if squat_state == "RECORDING":
                    current_squat_seq.append([knee_ang, hip_ang, torso_lean])
                    min_knee_seen = min(min_knee_seen, knee_ang)
                    lines.append((f"Allapot: GUGGOLAS MERESE ({len(current_squat_seq)} frame)", RED))

                    if knee_ang > END_KNEE_ANGLE:
                        squat_state = "IDLE"
                        if min_knee_seen < MIN_VALID_DEPTH and len(current_squat_seq) >= MIN_FRAMES:
                            squat_finished = True
                        else:
                            current_squat_seq = []
                            min_knee_seen = 180.0

            draw_panel(annotated, lines)

            db_cursor.execute('''
                INSERT INTO measurements (bal_kar_szog, jobb_kar_szog, bal_csuklo_gyorsulas, jobb_csuklo_gyorsulas)
                VALUES (?, ?, ?, ?)
            ''', (current_db_data["bal_kar_szog"], current_db_data["jobb_kar_szog"],
                  current_db_data["bal_csuklo_gyorsulas"], current_db_data["jobb_csuklo_gyorsulas"]))
            db_conn.commit()

        #Ha kész a guggolás, kiértékelés és a program megállítása ---
        if squat_finished:
            verdict, confidence, top_k = classify_squat(current_squat_seq, dataset, k=KNN_K)
            is_correct = (verdict == "Y")
            verdict_text = "HELYES GUGGOLAS (Y)" if is_correct else "HELYTELEN GUGGOLAS (N)"
            best_err, best_sim, best_sample = top_k[0]

            
            print("\n" + "=" * 58)
            print(f"EREDMENY: {verdict_text}")
            print(f"Dontes biztosaga (k={len(top_k)}): {confidence:.1f}% | Min. terdszog: {min_knee_seen:.1f} deg")
            print("Legkozelebbi tarolt mintak (DTW):")
            for rank, (err, sim, s) in enumerate(top_k, 1):
                note_str = f" | megjegyzes: {s['note']}" if s["note"] else ""
                print(f"  {rank}. {s['code']} -> hasonlosag: {sim:.1f}% (atlagos elteres: {err:.1f} deg){note_str}")
            print("=" * 58 + "\n")

            
            h, w, _ = annotated.shape
            box_color = (0, 150, 0) if is_correct else (0, 0, 180)
            cv2.rectangle(annotated, (20, h - 135), (w - 20, h - 20), box_color, -1)
            cv2.putText(annotated, verdict_text, (35, h - 92), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"Legkozelebbi: {best_sample['code']} ({best_sim:.1f}%, elt: {best_err:.1f} deg)",
                (35, h - 58), FONT, 0.6, WHITE, 2, cv2.LINE_AA
            )
            cv2.putText(annotated, "Nyomj meg barmilyen gombot a kilepeshez...",
                        (35, h - 30), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)

            cv2.imshow("AI Mozgaskoveto - Szog es Gyorsulas", annotated)
            cv2.waitKey(0)
            break

        cv2.imshow("AI Mozgaskoveto - Szog es Gyorsulas", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("r"): reset_all()

    cap.release()
    cv2.destroyAllWindows()
    db_conn.close()

if __name__ == "__main__":
    main()