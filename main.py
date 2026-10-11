import cv2
import time
import numpy as np
from ultralytics import YOLO

# a, b, c pontok(könyök, váll, csukló) x és y koordinátáinak alapján szögszámítás
def get_angle(a, b, c):
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]) #vízszintes tengelyhez képest bezárt irányszög számítás
    deg = np.abs(rad * 180.0 / np.pi) #Radiánból fokba váltja az értéket
    return 360.0 - deg if deg > 180.0 else deg

# két képkocka közötti szögváltozások számítása
def update_velocity(current_angle, prev_angle, dt, prev_velocity):
    if prev_angle is None or dt <= 0.001:
        return 0.0
    
    diff = current_angle - prev_angle
    # 1.5 fok alatti remegést zajnak tekintünk - NÖVELNI!!!
    raw_vel = 0.0 if abs(diff) < 1.5 else diff / dt
    # Simítás
    return 0.85 * prev_velocity + 0.15 * raw_vel

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# Jobb kar változói
r_prev_angle = None
r_velocity = 0.0

# Bal kar változói
l_prev_angle = None
l_velocity = 0.0

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    current_time = time.time()
    dt = current_time - prev_time

    #Pózadatok kinyerése és karok elemzése
    if len(results[0].keypoints.xy) > 0 and len(results[0].keypoints.xy[0]) > 10:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        conf = results[0].keypoints.conf[0].cpu().numpy()

        # Bal kar: Váll: 5, Könyök: 7, Csukló: 9 (ezek az indexek)
        if conf[5] > 0.5 and conf[7] > 0.5 and conf[9] > 0.5:
            l_angle = get_angle(kpts[5], kpts[7], kpts[9])
            l_velocity = update_velocity(l_angle, l_prev_angle, dt, l_velocity)
            l_prev_angle = l_angle

            # Szög kiírása a bal könyök mellé
            cv2.putText(annotated_frame, f"{int(l_angle)} deg", (int(kpts[7][0]), int(kpts[7][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            l_prev_angle = None
            l_velocity = 0.0

        # Jobb kar: Váll: 6, Könyök: 8, Csukló: 10 (ezek az indexek)
        if conf[6] > 0.5 and conf[8] > 0.5 and conf[10] > 0.5:
            r_angle = get_angle(kpts[6], kpts[8], kpts[10])
            r_velocity = update_velocity(r_angle, r_prev_angle, dt, r_velocity)
            r_prev_angle = r_angle

            # Szög kiírása a jobb könyök mellé
            cv2.putText(annotated_frame, f"{int(r_angle)} deg", (int(kpts[8][0]), int(kpts[8][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            r_prev_angle = None
            r_velocity = 0.0

    prev_time = current_time

    # kiírás a bal felső sarokba
    cv2.rectangle(annotated_frame, (10, 10), (320, 100), (30, 30, 30), -1)
    
    cv2.putText(annotated_frame, f"BAL kar vel:  {int(l_velocity)} deg/s", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"JOBB kar vel: {int(r_velocity)} deg/s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.imshow("Ketkaros Mozgaskoveto - PoT", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()