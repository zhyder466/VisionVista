import cv2
import easyocr
import threading
import playsound
from openai import OpenAI
import time
import warnings


client = OpenAI(api_key='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')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def extractText():
    threading.Thread(target=readText, args=("Extracting text from image",)).start()
    reader = easyocr.Reader(['en']) 
    image_path = 'image.jpg'
    result = reader.readtext(image_path)
    detected_text = ' '.join([detection[1] for detection in result])
    if len(detected_text) == 0:
        threading.Thread(target=readText, args=("No text detected from the image",)).start()
    else:
        threading.Thread(target=readText, args=(detected_text,)).start()

def readText(text):
    speech_file_path = 'extracted_text.mp3'

    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input= text
    )
    response.stream_to_file(speech_file_path)
    playsound.playsound(speech_file_path)

url = 'http://172.20.10.9:8080/video'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

def capture_and_save_image():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        return

    cv2.imshow('IP Camera', frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        file_path = 'image.jpg'

        cv2.imwrite(file_path, frame)
        print(f"Image saved as '{file_path}'")
        cap.release() 
        cv2.destroyAllWindows() 
        threading.Thread(target=readText, args=("Image successfully captured",)).start()
        time.sleep(3)
        threading.Thread(target=extractText, args=()).start()
        return False

    elif key == ord('q'):
        cap.release() 
        cv2.destroyAllWindows()
        return False

    return True

while True:
    if not capture_and_save_image():
        break