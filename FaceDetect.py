import cv2
import pickle
import face_recognition
import numpy as np
import threading
import time
import warnings
import playsound
from openai import OpenAI

client = OpenAI(api_key='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')
warnings.filterwarnings("ignore", category=DeprecationWarning)

url = 'http://172.20.10.4:8080/video'
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds

last_spoken_time = {}
cooldown_period = 5
confidence_threshold = 0.5

def generate_speech(class_name):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=f"I can see {class_name}."
    )
    response.stream_to_file('sound.mp3')
    playsound.playsound('sound.mp3')

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < confidence_threshold:
            name = studentIds[matchIndex].upper()
            current_time = time.time()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            if name not in last_spoken_time or (current_time - last_spoken_time[name] > cooldown_period):
                print(f"Confident match detected: {name} with face distance {faceDis[matchIndex]:.2f}")
                
                last_spoken_time[name] = current_time

                speech_thread = threading.Thread(target=generate_speech, args=(name,))
                speech_thread.start()

    cv2.imshow("Face", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
