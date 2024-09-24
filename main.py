import math
import time
import cv2
import cvzone
import pygame
from ultralytics import YOLO
import face_recognition  # Import the face_recognition library
import numpy as np  # Import numpy for array operations
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

confidence = 0.6

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("C:/Users/Muthukumar MSc/PycharmProjects/Antispoofing/models/l_version_1_300.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

# Load images of authorized persons and encode their faces
authorized_person_1_image = face_recognition.load_image_file("authorized_person_1.jpg")
authorized_person_1_encoding = face_recognition.face_encodings(authorized_person_1_image)[0]

authorized_person_2_image = face_recognition.load_image_file("authorized_person_2.jpg")
authorized_person_2_encoding = face_recognition.face_encodings(authorized_person_2_image)[0]

# Create a list of authorized face encodings and corresponding names
authorized_encodings = [authorized_person_1_encoding, authorized_person_2_encoding]
authorized_names = ["Muthukumar", "Person 2"]

# Load the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alert.wav")

# Email configuration
sender_email = "srimuthukumar786@gmail.com"
sender_password = "fhyr kfcu kwyp ogyr"
receiver_email = "srimuthukumar786@gmail.com"

def send_email(image):
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Unauthorized Person Detected"

    # Attach image
    img_data = cv2.imencode('.jpg', image)[1].tobytes()
    image = MIMEImage(img_data, name="intruder.jpg")
    msg.attach(image)

    # Send the message via SMTP
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    unauthorized_detected = False
    spoof_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                    # Perform face recognition on the detected face
                    face_image = img[y1:y2, x1:x2]
                    face_encodings = face_recognition.face_encodings(face_image)
                    if face_encodings:  # Check if any face is detected
                        face_encoding = face_encodings[0]  # Get the first face encoding
                        # Check if the detected face matches with any authorized person
                        authorized = False
                        for encoding, name in zip(authorized_encodings, authorized_names):
                            # Convert face encodings to numpy arrays
                            encoding = np.array(encoding)
                            face_encoding = np.array(face_encoding)
                            # Compare face encoding with authorized person's encoding
                            matches = face_recognition.compare_faces([encoding], face_encoding)
                            if matches[0]:
                                print(f"Authorized Person: {name}")
                                authorized = True
                                break
                        if not authorized:
                            # Unauthorized person detected
                            print("Unauthorized Person Detected!")
                            unauthorized_detected = True
                else:
                    color = (0, 0, 255)
                    spoof_detected = True

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    if unauthorized_detected or spoof_detected:
        alarm_sound.play()
        send_email(img)  # Send email with intruder image

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
