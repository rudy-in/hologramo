import cv2
import numpy as np
import random
import time

face_cascade = cv2.CascadeClassifier('../effects/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('../effects/haarcascade_smile.xml')
def hologram_effect(frame):
    overlay = frame.copy()

    # Converts to grayscale and apply green tint
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    green_tint = np.zeros_like(overlay)
    green_tint[:, :, 1] = gray  # Only green channel

    hologram = np.clip(green_tint.astype(np.int16), 0, 255).astype(np.uint8)

    
    for y in range(0, frame.shape[0], 3):
        hologram[y:y+1, :] = np.clip(hologram[y:y+1, :] - 40, 0, 255) # Adds up the scanlines (box)

    return hologram

def draw_hud(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (255, 255, 255), 1)
    cv2.putText(frame, "HOLOGRAPHIC INTERFACE", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, f"TIME: {timestamp}", (20, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

def detect_smile(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    expression = "No Face"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 0:
            expression = "Smiling"
            color = (0, 255, 0)
        else:
            expression = "Neutral"
            color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=3)

        label = expression
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - 30), (x + text_w + 10, y - 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), thickness=2)

    return expression

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        expression = detect_smile(gray, frame)

    
        holo = hologram_effect(frame) # calls for holographic effect 
        draw_hud(holo) 

        # Display expression text at the center (unstable)
        cv2.putText(holo, f"Expression: {expression}", (180, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Holographic Smile Scanner", holo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
