import cv2
from face_detect import detect_faces
from age_predict import predict_age
from smile_detect import detect_smile

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected")
    exit()

# store last predictions
last_age = None
last_smile = None
frame_count = 0

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # detect faces
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # ---- AGE (heavy → run rarely) ----
        if frame_count % 30 == 0:
            try:
                last_age = predict_age(face_img)
            except:
                last_age = None

        # ---- SMILE (medium → run often) ----
        if frame_count % 10 == 0:
            try:
                last_smile = detect_smile(face_img)
            except:
                last_smile = None

        # draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # ---- DISPLAY TEXT ----
    if last_age is not None:
        cv2.putText(frame, f"Age: {last_age}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if last_smile is not None:
        cv2.putText(frame, f"Smile: {last_smile}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Smile & Age Prediction", frame)

    # press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
