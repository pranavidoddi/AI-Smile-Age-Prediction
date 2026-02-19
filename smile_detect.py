import cv2

# load smile cascade
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

def detect_smile(face):
    if face is None:
        return False

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.7,
        minNeighbors=22,
        minSize=(25, 25)
    )

    if len(smiles) > 0:
        return True
    else:
        return False
