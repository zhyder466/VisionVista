import cv2
import pickle
import face_recognition
import numpy as np
import threading
import time
import warnings
import math
import playsound
import os
import pyaudio
import wave
import requests
from openai import OpenAI
from ultralytics import YOLO

# Initialize OpenAI client
client = OpenAI(api_key='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')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Video stream URL
url = 'http://172.20.10.3:8080/video'
url2 = 'http://10.102.128.138:8080/video'


cap = cv2.VideoCapture(url) 
cap.set(3, 1280)
cap.set(4, 720)

# Load known face encodings and IDs
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Define class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Define center point and pixel to meter ratio for distance calculation
center_point = (1280 // 2, 720 // 2)
pixel_per_meter = 120

# Initialize variables for speech cooldown and frame processing
last_spoken_time = {}
cooldown_period = 5
confidence_threshold = 0.5 
frame_skip = 3  # Process every 3rd frame to reduce load
frame_count = 0  # Counter for frame skipping

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the distance of the object from the center point.
    """
    x_centroid = (x1 + x2) // 2
    y_centroid = (y1 + y2) // 2
    distance = math.sqrt((x_centroid - center_point[0]) ** 2 + (y_centroid - center_point[1]) ** 2) / pixel_per_meter
    return distance

def generate_speech(message):
    """
    Generate speech from the given message using OpenAI's API and play it.
    """
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

def clear_buffer(cap):
    """
    Clear the video capture buffer to reduce latency.
    """
    for _ in range(5):
        cap.grab()  # Skip a few frames to ensure buffer is clear

def voice_command_listener():
    """
    Listen for voice commands and execute actions based on the recognized text.
    """
    # Set your OpenAI API key directly here
    API_KEY = "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"

    # Audio recording parameters
    CHUNK = 2048  # Increased buffer size to prevent overflow
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 16000  # Audio sample rate
    RECORD_SECONDS = 1  # Reduced duration for faster response
    WAVE_OUTPUT_FILENAME = "/Users/hyder/Downloads/VisionVista/Res/output.wav"  # Temporary file for audio

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Voice command listener started. Say 'exit' to terminate the program.")

    def record_audio():
        """
        Record audio for a specified duration and save it to a WAV file.
        """
        frames = []
        try:
            # Record audio for the specified duration
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)  # Handle buffer overflow
                frames.append(data)
        except IOError as e:
            print(f"Error recording audio: {e}")
            return None

        # Save the recorded audio as a .wav file
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return WAVE_OUTPUT_FILENAME

    def transcribe_audio(audio_file):
        """
        Transcribe the given audio file using OpenAI's Whisper API.
        """
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        files = {
            'file': open(audio_file, 'rb'),
            'model': (None, 'whisper-1'),
        }

        try:
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            return response.json().get('text', '').strip().lower()
        except requests.exceptions.RequestException as e:
            print(f"Error transcribing audio: {e}")
            return None

    try:
        while True:
            audio_file = record_audio()
            if not audio_file:
                continue

            print("Transcribing audio...")
            transcript = transcribe_audio(audio_file)
            if transcript:
                print(f"Transcribed Text: {transcript}")
                # Check for 'exit' or its variations
                if "exit" in transcript or "exiting" in transcript or "stop" in transcript:
                    print("Exit command received. Terminating the program.")
                    os._exit(0)  # Forcefully terminate the program

            # No sleep to make it listen continuously
    except KeyboardInterrupt:
        print("Voice command listener stopped by user.")
    finally:
        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Start the voice command listener in a separate thread
voice_thread = threading.Thread(target=voice_command_listener, daemon=True)
voice_thread.start()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from video stream. Exiting.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    clear_buffer(cap)

    # Resize and convert the image for face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
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
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to original image size

            distance = calculate_distance(x1, y1, x2, y2)

            # Draw rectangle and name on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Handle speech cooldown
            if name not in last_spoken_time or (current_time - last_spoken_time[name] > cooldown_period):
                print(f"Confident match detected: {name} with face distance {faceDis[matchIndex]:.2f}")
                
                last_spoken_time[name] = current_time

                # Generate speech in a separate thread
                threading.Thread(target=generate_speech, args=(f"I see {name} {distance:.2f} meters away.",)).start()

            known_person_detected = True

    # Perform object detection using YOLO
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

                # Draw rectangle and label on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, f"{class_name} ({confidence:.2f}), {distance:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if class_name == "person" and not known_person_detected:
                    current_time = time.time()

                    if "unknown person" not in last_spoken_time or (current_time - last_spoken_time["unknown person"] > cooldown_period):
                        last_spoken_time["unknown person"] = current_time
                        threading.Thread(target=generate_speech, args=("Unknown person detected.",)).start()

                elif class_name != "person":  
                    current_time = time.time()
                    if class_name not in last_spoken_time or (current_time - last_spoken_time[class_name] > cooldown_period):
                        last_spoken_time[class_name] = current_time
                        threading.Thread(target=generate_speech, args=(f"I see a {class_name} {distance:.2f} meters away.",)).start()

    # Display the resulting image
    cv2.imshow("Face & Object Detection", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit command received via keyboard. Terminating the program.")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
