import cv2
import os
import re
import json
import time
import math
import urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np
import mediapipe as mp


MP_MODEL_PATH = Path("pose_landmarker_full.task")
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DATASET_DIR = "squat_dataset_mp"

MIRROR_VIEW = True
MIN_CONFIDENCE = 0.6
MIN_DT = 0.001
MAX_DT = 0.5
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0

# Guggolás-szegmentálási küszöbök (fokban)
START_KNEE_ANGLE = 155.0      # Ennél kisebb térdszögnél indul a felvétel
END_KNEE_ANGLE = 162.0        # Ennél nagyobb térdszögnél zárul le (hiszterézis)
MIN_DEPTH_TRIGGER = 148.0     # Legalább ennyire hajoljon be (a túl sekély 'N' minták is beleférnek)
MIN_SQUAT_FRAMES = 10         # Túl rövid rándulások kiszűrése
MAX_MISSING_FRAMES = 12       # Ennyi képkockányi takarás után megszakítja az aktuális felvételt
POST_SAVE_COOLDOWN = 3.0      # Mentés után ennyi mp van visszasétálni a kamera elé

SQUAT_SIDES = [
    ("left",  (11, 23, 25, 27, 29, 31)),
    ("right", (12, 24, 26, 28, 30, 32)),
]

POSE_CONNECTIONS = (
    (11, 12), (11, 23), (12, 24), (23, 24),  # Törzs
    (11, 13), (13, 15), (12, 14), (14, 16),  # Karok
    (23, 25), (25, 27), (24, 26), (26, 28),  # Lábak
    (27, 29), (29, 31), (27, 31),            # Bal lábfej
    (28, 30), (30, 32), (28, 32),            # Jobb lábfej
)
DRAW_KEYPOINTS = sorted({i for edge in POSE_CONNECTIONS for i in edge})

ALL_KEYPOINTS = list(range(33))
FONT = cv2.FONT_HERSHEY_SIMPLEX


# MediaPipe kinyerés és csontváz rajzolás
def create_pose_landmarker():
    if not MP_MODEL_PATH.exists():
        print(f"MediaPipe modell letoltese ({MP_MODEL_PATH.name})...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        print("Letoltes kesz.")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def extract_mp_person(result, frame_w, frame_h):
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    points = np.zeros((33, 2), dtype=float)
    confs = np.zeros(33, dtype=float)

    for i, lm in enumerate(lms):
        points[i] = (lm.x * frame_w, lm.y * frame_h)
        vis = float(lm.visibility) if lm.visibility is not None else 1.0
        # Képből kilógó pontok kiszűrése
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            vis = 0.0
        confs[i] = vis

    visible_y = points[confs >= MIN_CONFIDENCE, 1]
    body_h = float(visible_y.max() - visible_y.min()) if len(visible_y) >= 2 else None
    return points, confs, body_h


def draw_skeleton(frame, points, confs):
    for i, j in POSE_CONNECTIONS:
        if confs[i] >= MIN_CONFIDENCE and confs[j] >= MIN_CONFIDENCE:
            p1 = (int(points[i, 0]), int(points[i, 1]))
            p2 = (int(points[j, 0]), int(points[j, 1]))
            cv2.line(frame, p1, p2, (255, 180, 0), 2, cv2.LINE_AA)
    for i in DRAW_KEYPOINTS:
        if confs[i] >= MIN_CONFIDENCE:
            cv2.circle(frame, (int(points[i, 0]), int(points[i, 1])), 4, (0, 255, 255), -1)


# Geometria és szűrő (azonos a főprogrammal)
def get_angle(a, b, c):
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / np.pi)
    return 360.0 - deg if deg > 180.0 else deg


def get_torso_lean(shoulder, hip):
    """A törzs dőlésszöge a függőlegeshez képest (0 deg = teljesen egyenes)."""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    cos_val = max(-1.0, min(1.0, -dy / norm))
    return math.degrees(math.acos(cos_val))


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


def extract_squat_metrics(fp):
    """Kinyeri a térd-, csípő-, törzsdőlés- és bokaszöget a látható oldal(ak)ról."""
    valid = []
    valid_ankles = []
    for _, (sh, hip, knee, ankle, heel, foot) in SQUAT_SIDES:
        if all(fp.get(idx) is not None for idx in (sh, hip, knee, ankle)):
            knee_ang = get_angle(fp[hip], fp[knee], fp[ankle])
            hip_ang = get_angle(fp[sh], fp[hip], fp[knee])
            torso_lean = get_torso_lean(fp[sh], fp[hip])
            valid.append((knee_ang, hip_ang, torso_lean))
            if fp.get(foot) is not None:
                valid_ankles.append(get_angle(fp[knee], fp[ankle], fp[foot]))
    if not valid:
        return None
    ankle_ang = float(np.mean(valid_ankles)) if valid_ankles else None
    return (
        float(np.mean([v[0] for v in valid])),
        float(np.mean([v[1] for v in valid])),
        float(np.mean([v[2] for v in valid])),
        ankle_ang,
    )


def draw_panel(frame, lines):
    height = 14 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, 10 + height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    y = 35
    for text, color in lines:
        cv2.putText(frame, text, (20, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 24


# Fájlkezelés és parancssori címkézés
def get_next_id(folder):
    os.makedirs(folder, exist_ok=True)
    max_id = 0
    for fname in os.listdir(folder):
        match = re.match(r"^(\d+)_[YN]\.json$", fname)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def build_summary(frames):
    knee_vals = [f["knee_angle"] for f in frames]
    hip_vals = [f["hip_angle"] for f in frames]
    torso_vals = [f["torso_lean"] for f in frames]
    ankle_vals = [f["ankle_angle"] for f in frames if f.get("ankle_angle") is not None]
    return {
        "frame_count": len(frames),
        "duration_sec": round(frames[-1]["t"], 3),
        "min_knee_angle": round(min(knee_vals), 2),
        "min_hip_angle": round(min(hip_vals), 2),
        "max_torso_lean": round(max(torso_vals), 2),
        "min_ankle_angle": round(min(ankle_vals), 2) if ankle_vals else None,
    }


def prompt_label(squat_id, summary):
    ankle_str = f"{summary['min_ankle_angle']:.1f} deg" if summary["min_ankle_angle"] is not None else "n/a"
    print("\n" + "=" * 54)
    print(f"Guggolas #{squat_id:03d} rogzitve!")
    print(f"  Idotartam:      {summary['duration_sec']:.2f} s ({summary['frame_count']} frame)")
    print(f"  Min. terdszog:  {summary['min_knee_angle']:.1f} deg")
    print(f"  Min. csiposzog: {summary['min_hip_angle']:.1f} deg")
    print(f"  Max. torzsdoles:{summary['max_torso_lean']:.1f} deg")
    print(f"  Min. bokaszog:  {ankle_str}")
    print("-" * 54)

    while True:
        raw = input("Helyes volt? [y=Igen / n=Nem / d=Eldobas] (pl. 'n sekely'): ").strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        choice = parts[0].upper()
        note = parts[1].strip() if len(parts) > 1 else ""
        if choice in ("Y", "N", "D"):
            return choice, note
        print("Ervenytelen valasz. Hasznald a 'y', 'n' vagy 'd' betut!")


def save_squat_json(folder, squat_id, label, note, summary, frames):
    code = f"{squat_id:03d}_{label}"
    filepath = os.path.join(folder, f"{code}.json")
    payload = {
        "code": code,
        "id": squat_id,
        "label": label,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "frames": frames,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath


# Főprogram
def main():
    landmarker = create_pose_landmarker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Nem sikerult megnyitni a kamerat.")
        landmarker.close()
        return

    next_id = get_next_id(DATASET_DIR)
    kp_filters = {
        i: OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        for i in ALL_KEYPOINTS
    }

    def reset_filters():
        for f in kp_filters.values():
            f.reset()

    state = "WAIT_STAND"   # WAIT_STAND -> READY -> RECORDING
    paused = False
    recorded_frames = []
    record_start_t = 0.0
    min_knee_seen = 180.0
    missing_count = 0
    cooldown_until = 0.0
    last_ts_ms = -1
    last_saved_msg = "Meg nincs mentett minta ebben a futasban"

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    GRAY = (180, 180, 180)

    print(f"Guggolas rogzito (MediaPipe) elinditva. Mentesi mappa: ./{DATASET_DIR}/")
    print(f"Kovetkezo azonosito: {next_id:03d} | Billentyuk: 'p' = szunet, 'q' = kilepes")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]

        # MediaPipe PoseLandmarker futtatása VIDEO módban
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(now * 1000)
        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms

        mp_result = landmarker.detect_for_video(mp_image, ts_ms)
        annotated = frame.copy()
        person = extract_mp_person(mp_result, frame_w, frame_h)
        if person is not None:
            draw_skeleton(annotated, person[0], person[1])

        lines = [
            (f"Kovetkezo ID: {next_id:03d} | 'p': szunet | 'q': kilepes", GRAY),
            (last_saved_msg, WHITE),
        ]

        if paused:
            lines.append(("SZUNETELTETVE (nyomj 'p'-t a folytatashoz)", YELLOW))
            draw_panel(annotated, lines)
            cv2.imshow("Guggolas Rogzito (MediaPipe)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = False
                state = "WAIT_STAND"
                recorded_frames = []
                reset_filters()
            continue

        if now < cooldown_until:
            rem = cooldown_until - now
            lines.append((f"Keszulj! Allj pozicioba: {rem:.1f} s", YELLOW))
            draw_panel(annotated, lines)
            cv2.imshow("Guggolas Rogzito (MediaPipe)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        metrics = None
        fp = {}
        body_h = None

        if person is not None:
            points, confs, body_h = person
            for i in ALL_KEYPOINTS:
                if confs[i] >= MIN_CONFIDENCE:
                    fp[i] = kp_filters[i](points[i], now)
                else:
                    kp_filters[i].reset()
                    fp[i] = None
            metrics = extract_squat_metrics(fp)
        else:
            reset_filters()

        if metrics is None:
            if state == "RECORDING":
                missing_count += 1
                lines.append((f"FELVETEL - pont elveszett ({missing_count}/{MAX_MISSING_FRAMES})", RED))
                if missing_count >= MAX_MISSING_FRAMES:
                    state = "WAIT_STAND"
                    recorded_frames = []
                    missing_count = 0
                    last_saved_msg = "Felvetel megszakitva (kiesett a kepbol)"
            else:
                lines.append(("Allj oldalra ugy, hogy vall-csipo-terd-boka latszodjon", RED))
        else:
            missing_count = 0
            knee_ang, hip_ang, torso_lean, ankle_ang = metrics
            ankle_txt = f"{int(ankle_ang)} deg" if ankle_ang is not None else "-"
            lines.append((
                f"Terd: {int(knee_ang)} | Csipo: {int(hip_ang)} | Torzs: {int(torso_lean)} | Boka: {ankle_txt}",
                YELLOW,
            ))

            def make_frame_entry(t_rel):
                kpt_dict = {}
                for idx in ALL_KEYPOINTS:
                    if fp[idx] is not None:
                        kpt_dict[str(idx)] = [
                            round(float(fp[idx][0]), 2),
                            round(float(fp[idx][1]), 2),
                            round(float(confs[idx]), 3),
                        ]
                return {
                    "t": round(t_rel, 4),
                    "knee_angle": round(knee_ang, 2),
                    "hip_angle": round(hip_ang, 2),
                    "torso_lean": round(torso_lean, 2),
                    "ankle_angle": round(ankle_ang, 2) if ankle_ang is not None else None,
                    "body_h": round(body_h, 2) if body_h is not None else None,
                    "keypoints": kpt_dict,
                }

            if state == "WAIT_STAND":
                lines.append(("Allj egyenesen a kiindulo helyzethez...", WHITE))
                if knee_ang >= END_KNEE_ANGLE:
                    state = "READY"

            elif state == "READY":
                lines.append(("KESZ - Indithatod a guggolast!", GREEN))
                if knee_ang < START_KNEE_ANGLE:
                    state = "RECORDING"
                    record_start_t = now
                    min_knee_seen = knee_ang
                    recorded_frames = [make_frame_entry(0.0)]

            elif state == "RECORDING":
                recorded_frames.append(make_frame_entry(now - record_start_t))
                min_knee_seen = min(min_knee_seen, knee_ang)
                lines.append((f"ROGZITES FOLYAMATBAN... ({len(recorded_frames)} frame)", RED))

                if knee_ang >= END_KNEE_ANGLE:
                    state = "WAIT_STAND"
                    if min_knee_seen <= MIN_DEPTH_TRIGGER and len(recorded_frames) >= MIN_SQUAT_FRAMES:
                        summary = build_summary(recorded_frames)

                        draw_panel(annotated, [
                            ("GUGGOLAS KESZ - VALASZOLJ A KONZOLBAN!", YELLOW),
                            ("y = Helyes | n = Helytelen | d = Eldobas", WHITE),
                        ])
                        cv2.imshow("Guggolas Rogzito (MediaPipe)", annotated)
                        cv2.waitKey(1)

                        choice, note = prompt_label(next_id, summary)
                        if choice in ("Y", "N"):
                            saved_path = save_squat_json(
                                DATASET_DIR, next_id, choice, note, summary, recorded_frames
                            )
                            last_saved_msg = f"Elmentve: {os.path.basename(saved_path)}"
                            print(f"Mentve: {saved_path}")
                            next_id += 1
                        else:
                            last_saved_msg = "Utolso guggolas eldobva."
                            print("Eldobva.")

                        for _ in range(5):
                            cap.grab()
                        reset_filters()
                        cooldown_until = time.perf_counter() + POST_SAVE_COOLDOWN
                    else:
                        last_saved_msg = "Tul rovid/sekely mozdulat - figyelmen kivul hagyva"

                    recorded_frames = []

        draw_panel(annotated, lines)
        cv2.imshow("Guggolas Rogzito (MediaPipe)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = True

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()