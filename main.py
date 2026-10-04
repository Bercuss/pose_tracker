import cv2
from ultralytics import YOLO

# Betölti az előtanított könnyű pózbecslő modellt (első futáskor automatikusan letölti)
model = YOLO("yolov8n-pose.pt")

# Webkamera indítása
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kép tükrözése
    frame = cv2.flip(frame, 1)

    # Kulcspontok becslése és csontváz rárajzolása
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("AI Mozgaskoveto - PoT", annotated_frame)

    # Kilépés 'q' gombbal
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()