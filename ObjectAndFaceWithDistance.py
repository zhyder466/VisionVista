import cv2
import pickle
import face_recognition
import numpy as np
import threading
import time
import warnings
import math
import playsound
from openai import OpenAI
from ultralytics import YOLO
import os

client = OpenAI(api_key='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')
warnings.filterwarnings("ignore", category=DeprecationWarning)
url = 'http://172.20.10.5:8080/video'
cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 720)
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

center_point = (1280 // 2, 720 // 2)
pixel_per_meter = 120

last_spoken_time = {}
cooldown_period = 5
confidence_threshold = 0.5 

def calculate_distance(x1, y1, x2, y2):
    x_centroid = (x1 + x2) // 2
    y_centroid = (y1 + y2) // 2
    distance = math.sqrt((x_centroid - center_point[0]) ** 2 + (y_centroid - center_point[1]) ** 2) / pixel_per_meter
    return distance

def generate_speech(message):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=message
        )
        response.stream_to_file('/Users/hyder/Downloads/VisionVista/Res/sound.mp3')
        playsound.playsound('/Users/hyder/Downloads/VisionVista/Res/sound.mp3')
    except Exception as e:
        print(f"Error generating speech: {e}")

while True:
    success, img = cap.read()
    if not success:
        break
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    known_person_detected = False

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < confidence_threshold:
            name = studentIds[matchIndex].upper()
            current_time = time.time()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            distance = calculate_distance(x1, y1, x2, y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            if name not in last_spoken_time or (current_time - last_spoken_time[name] > cooldown_period):
                print(f"Confident match detected: {name} with face distance {faceDis[matchIndex]:.2f}")
                
                last_spoken_time[name] = current_time

                speech_thread = threading.Thread(target=generate_speech, args=(f"I see {name} {distance:.2f} meters away.",))
                speech_thread.start()

            known_person_detected = True

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                distance = calculate_distance(x1, y1, x2, y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, f"{class_name} ({confidence:.2f}), {distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if class_name == "person" and not known_person_detected:
                    current_time = time.time()

                    if "unknown person" not in last_spoken_time or (current_time - last_spoken_time["unknown person"] > cooldown_period):
                        last_spoken_time["unknown person"] = current_time

                        speech_thread = threading.Thread(target=generate_speech, args=("Unknown person detected.",))
                        speech_thread.start()
                elif class_name != "person":  
                    current_time = time.time()
                    if class_name not in last_spoken_time or (current_time - last_spoken_time[class_name] > cooldown_period):
                        last_spoken_time[class_name] = current_time
                        speech_thread = threading.Thread(target=generate_speech, args=(f"I see a {class_name} {distance:.2f} meters away.",))
                        speech_thread.start()

    cv2.imshow("Face & Object Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        os._exit()

cap.release()
cv2.destroyAllWindows()
