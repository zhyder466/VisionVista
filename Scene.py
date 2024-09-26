import cv2
import threading
import time
import requests
import base64
import playsound
import os
import sounddevice as sd
import numpy as np
import wave

api_key = "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"

CHUNK = 1024
FORMAT = np.int16 
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "/Users/hyder/Downloads/VisionVista/Res/command2.wav"

pause_listening_event = threading.Event()
url = 'http://172.20.10.3:8080/video'
url2 = 'http://10.102.128.138:8080/video'

mode = "initial"
current_image_path = None

def capture_image_from_camera(command_event):
    global mode, current_image_path
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        cv2.imshow('Camera', frame)

        if command_event.is_set() and mode == "initial":
            pause_listening_event.clear()

            current_image_path = "/Users/hyder/Downloads/VisionVista/Res/captured_image2.jpg"
            cv2.imwrite(current_image_path, frame)
            print(f"Image saved as '{current_image_path}'")

            text_to_speech("Image successfully captured. Let me explain what's in this image.")
            process_image_and_read(current_image_path)

            mode = "interactive"

            ask_for_further_questions()

            command_event.clear()

            pause_listening_event.set()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_audio():
    print("Listening for voice command...")
    
    # Record the audio using sounddevice
    recording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
    sd.wait()  # Wait until recording is finished

    # Convert the recorded data to bytes
    frames = recording.astype(np.int16).tobytes()

    # Save to a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for FORMAT int16
        wf.setframerate(RATE)
        wf.writeframes(frames)

    return WAVE_OUTPUT_FILENAME

def transcribe_audio(audio_file):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {
        'file': open(audio_file, 'rb'),
        'model': (None, 'whisper-1'),
    }

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()['text']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def listen_for_command(command_event):
    global mode
    while True:
        pause_listening_event.wait()

        audio_file = record_audio()
        user_command = transcribe_audio(audio_file)

        if user_command:
            print(f"User said: {user_command}")

            if mode == "initial":
                if "capture" in user_command.lower():
                    command_event.set()

                elif "exit" in user_command.lower():
                    text_to_speech("Ok, exiting the scenario description mode.")
                    os._exit(0)

                else:
                    text_to_speech("I don't understand. Please say 'capture' to take a picture or 'exit' to leave.")

            elif mode == "interactive":
                if "exit" in user_command.lower():
                    text_to_speech("Ok, exiting the scenario description mode.")
                    os._exit(0)
                else:
                    process_user_query(user_command)

def process_image_and_read(image_path):
    print(f"Processing the image: {image_path}")
    
    description = get_openai_response(image_path, "What's in this image?")
    
    if description:
        print(f"Extracted Description: {description}")
        text_to_speech(description)
    else:
        print("No description found.")
        text_to_speech("I'm unable to describe the image.")

def ask_for_further_questions():
    text_to_speech("Do you have any further questions about this image? You can ask a question or say exit to close the program.")
    pause_listening_event.set()

def process_user_query(user_query):
    global current_image_path
    if current_image_path:
        print(f"User Query: {user_query}")
        response = get_openai_response(current_image_path, user_query)

        if response:
            print(f"Response: {response}")
            text_to_speech(response)
        else:
            print("No response received.")
            text_to_speech("Sorry, I couldn't find an answer to your question.")
    else:
        text_to_speech("No image available for reference. Please capture an image first.")

    ask_for_further_questions()

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    return None

def get_openai_response(image_path, user_query):
    base64_image = encode_image(image_path) if image_path else None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if base64_image:
        content = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    else:
        content = [{"type": "text", "text": user_query}]

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
    return None

def text_to_speech(text):
    speech_file_path = '/Users/hyder/Downloads/VisionVista/Res/response_text2.mp3'

    tts_payload = {
        "model": "tts-1",
        "voice": "onyx",
        "input": text
    }

    tts_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    tts_response = requests.post("https://api.openai.com/v1/audio/speech", headers=tts_headers, json=tts_payload)

    with open(speech_file_path, 'wb') as audio_file:
        audio_file.write(tts_response.content)

    playsound.playsound(speech_file_path)

    os.remove(speech_file_path)

def welcome_message():
    text_to_speech("Welcome to the scenario description mode. Say capture to capture the image or exit to close the program.")

if __name__ == "__main__":
    welcome_message()

    command_event = threading.Event()

    voice_thread = threading.Thread(target=listen_for_command, args=(command_event,))
    voice_thread.start()

    pause_listening_event.set()

    capture_image_from_camera(command_event)
