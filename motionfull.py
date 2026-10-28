import cv2
import time
import math
import numpy as np
from collections import deque
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Beállítások
# ----------------------------------------------------------------------
# n = leggyorsabb, s/m = pontosabb és kevésbé remeg (de lassabb, főleg CPU-n)
MODEL_NAME = "yolov8n-pose.pt"

# A kép tükrözve van (cv2.flip), a YOLO viszont nem tükrözött képeken tanult,
# ezért a "bal/jobb" indexeket meg kell cserélni. Ha mégis fordítva látod: False.
MIRROR_VIEW = True

MIN_CONFIDENCE = 0.6      # ennél kisebb megbízhatóságú kulcspontot eldobunk
MIN_DT = 0.001            # (s)
MAX_DT = 0.5              # ennél hosszabb kihagyás után újraindítjuk a mérést (s)

# --- One Euro szűrő a kulcspontokra (pixel koordináták) ---
# Ha állva még remeg a pont:  csökkentsd a MIN_CUTOFF-ot (pl. 0.5)
# Ha mozgásnál késik/csúszik: növeld a BETA-t (pl. 0.05)
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0

# --- Időablakok a sebesség / gyorsulás számításához (másodperc) ---
VEL_WINDOW = 0.12         # ennyi idő alatti elmozdulásból számolunk sebességet
ACC_WINDOW = 0.20         # ennyi idő alatti sebességváltozásból gyorsulást

# --- Zajküszöbök ---
ANGLE_VEL_FLOOR = 6.0     # deg/s alatt a szögsebességet 0-nak vesszük
ACC_FLOOR = 150.0         # px/s^2 alatt a gyorsulást 0-nak vesszük
ANGLE_SMOOTH = 0.4        # szögsebesség simítás: az új minta súlya (0..1)
ACC_SMOOTH = 0.4          # gyorsulás simítás: az új minta súlya (0..1)

# Ha True, a px értékek a testmagassághoz vannak viszonyítva, így nem függnek
# attól, milyen messze állsz a kamerától.
NORMALIZE_BY_BODY = False


def side(left_idx, right_idx):
    """Tükrözött nézetnél a YOLO bal/jobb pontja felcserélődik."""
    return (right_idx, left_idx) if MIRROR_VIEW else (left_idx, right_idx)


# YOLOv8 COCO kulcspont-indexek: (váll, könyök, csukló)
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

USED_KEYPOINTS = sorted({i for t in ARMS.values() for i in t} | set(JOINTS.values()))

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------
# Szögszámítás (a, b, c: váll, könyök, csukló)
# ----------------------------------------------------------------------
def get_angle(a, b, c):
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / np.pi)
    return 360.0 - deg if deg > 180.0 else deg


# ----------------------------------------------------------------------
# One Euro szűrő: állva erősen simít, gyors mozgásnál alig késik
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
# Időablakos derivált: (mostani - ~window ideje mért) / eltelt idő
# ----------------------------------------------------------------------
class WindowBuffer:
    def __init__(self, window):
        self.window = window
        self.buf = deque()

    def push(self, t, v):
        self.buf.append((t, v))
        # a legrégebbi elem az legyen, ami még legalább 'window' idős
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
# Szögsebesség követő (egy kar)
# ----------------------------------------------------------------------
class AngleTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buf = WindowBuffer(VEL_WINDOW)
        self.last_time = None
        self.velocity = 0.0  # fok/s
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
# Ízület sebesség + gyorsulás követő (egy már szűrt kulcspont)
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
# Segédfüggvények
# ----------------------------------------------------------------------
def pick_person(result):
    """A legnagyobb észlelt személy kulcspontjai, megbízhatóságai és magassága."""
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
    if result.keypoints.conf is not None:
        confs = result.keypoints.conf[idx].cpu().numpy()
    else:
        confs = np.ones(len(points))

    return points, confs, body_h


def draw_panel(frame, lines):
    """Félig átlátszó háttér + szöveges sorok a bal felső sarokban."""
    height = 14 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 10 + height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 35
    for text, color in lines:
        cv2.putText(frame, text, (20, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 24


# ----------------------------------------------------------------------
# Főprogram
# ----------------------------------------------------------------------
def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nem sikerult megnyitni a kamerat.")
        return

    kp_filters = {
        i: OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        for i in USED_KEYPOINTS
    }
    angle_trackers = {name: AngleTracker() for name in ARMS}
    point_trackers = {name: PointTracker() for name in JOINTS}
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        result = model(frame, verbose=False)[0]
        annotated = result.plot()

        person = pick_person(result)

        if person is None:
            reset_all()
            cv2.putText(annotated, "Nem lathato ember", (20, 35),
                        FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            points, confs, body_h = person
            lines = []

            # ---- 0) Kulcspontok szűrése ----
            fp = {}
            for i in USED_KEYPOINTS:
                if confs[i] >= MIN_CONFIDENCE:
                    fp[i] = kp_filters[i](points[i], now)
                else:
                    kp_filters[i].reset()
                    fp[i] = None

            # Skála (testmagasság simítva, hogy ne ugráljon)
            scale = 1.0
            if NORMALIZE_BY_BODY and body_h:
                body_h_ema = body_h if body_h_ema is None else 0.9 * body_h_ema + 0.1 * body_h
                scale = body_h_ema
            acc_floor = ACC_FLOOR / scale

            # ---- 1) Karok: könyökszög + szögsebesség ----
            lines.append(("--- Karok (szog, szogsebesseg) ---", GRAY))
            arm_colors = {"BAL kar": GREEN, "JOBB kar": ORANGE}

            for name, (s, e, w) in ARMS.items():
                tracker = angle_trackers[name]
                if fp[s] is not None and fp[e] is not None and fp[w] is not None:
                    angle = get_angle(fp[s], fp[e], fp[w])
                    tracker.update(angle, now)

                    cv2.putText(annotated, f"{int(angle)} deg",
                                (int(fp[e][0]), int(fp[e][1])),
                                FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
                    lines.append((f"{name}: {int(angle)} deg  {int(tracker.velocity):+d} deg/s",
                                  arm_colors[name]))
                else:
                    tracker.reset()
                    lines.append((f"{name}: nem lathato", GRAY))

            # ---- 2) Ízületek: gyorsulás ----
            lines.append((f"--- Izuletek (gyorsulas, {unit}) ---", GRAY))

            for name, idx in JOINTS.items():
                tracker = point_trackers[name]
                p = fp[idx]
                if p is None:
                    tracker.reset()
                    lines.append((f"{name}: nem lathato", GRAY))
                    continue

                # a szűrt pont kirajzolása (ezt használjuk a méréshez)
                cv2.circle(annotated, (int(p[0]), int(p[1])), 5, YELLOW, -1)

                tracker.update((p[0] / scale, p[1] / scale), now, acc_floor)

                if tracker.acceleration is not None:
                    v = tracker.acceleration
                    text = f"{name}: {v:.2f} {unit}" if NORMALIZE_BY_BODY \
                        else f"{name}: {v:.0f} {unit}"
                    lines.append((text, GREEN))
                elif tracker.speed is not None:
                    lines.append((f"{name}: meres...", YELLOW))
                else:
                    lines.append((f"{name}: varakozas...", WHITE))

            draw_panel(annotated, lines)

        cv2.imshow("AI Mozgaskoveto - Szog es Gyorsulas", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            reset_all()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
