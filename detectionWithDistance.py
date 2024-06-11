import speech_recognition as sr
from ultralytics import YOLO
import cv2
import math
from openai import OpenAI
import playsound
import threading
import time
import warnings

client = OpenAI(api_key='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')
warnings.filterwarnings("ignore", category=DeprecationWarning)

speech_file_path = "speech.mp3"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolov8s.pt")

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


center_point = (320, 240)
pixel_per_meter = 35 

last_spoken = {}
last_announcement_time = 0
announcement_cooldown = 5 

def generate_speech(class_name, distance):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=f"I can see a {class_name} at a distance of {distance:.2f} meters."
    )
    response.stream_to_file(speech_file_path)
    playsound.playsound(speech_file_path)

def can_speak(class_name):
    last_time_spoken = last_spoken.get(class_name, 0)
    current_time = time.time()
    if current_time - last_time_spoken >= announcement_cooldown: 
        last_spoken[class_name] = current_time
        return True
    return False

def handle_voice_command(command):
    if "exit" in command:
        cv2.destroyAllWindows()
        cap.release()
        exit()

def process_frame(img):
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            x_centroid, y_centroid = (x1 + x2) // 2, (y1 + y2) // 2

            distance = math.sqrt((x_centroid - center_point[0]) ** 2 + (y_centroid - center_point[1]) ** 2) / pixel_per_meter
            confidence = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            class_name = classNames[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cv2.putText(img, f"{class_name} ({confidence:.2f}), {distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if confidence >= 0.5 and can_speak(class_name):
                threading.Thread(target=generate_speech, args=(class_name, distance)).start()

    cv2.imshow('Webcam', img)

def listen_for_commands():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice commands...")
        while True:
            audio = recognizer.listen(source)

            try:
                command = recognizer.recognize_google(audio).lower()
                print("Command:", command)
                handle_voice_command(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

threading.Thread(target=listen_for_commands).start()

while True:
    success, img = cap.read()
    if not success:
        break
    
    process_frame(img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
