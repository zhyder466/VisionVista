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
url = 'http://172.20.10.2:8080/video'
url2 = 'http://10.102.128.138:8080/video'


CHUNK = 1024  # Buffer size
FORMAT = 'int16'  # Audio format
CHANNELS = 1  # Mono input
RATE = 44000  # Sample rate (16kHz is optimal for voice recognition)
RECORD_SECONDS = 5  # Duration of audio capture for each request
WAVE_OUTPUT_FILENAME = "/Users/hyder/Downloads/VisionVista/Res/command1.wav"

pause_listening_event = threading.Event()

def capture_image_from_camera(command_event):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        cv2.imshow('Camera', frame)

        if command_event.is_set():
            pause_listening_event.clear()  # Clear the pause to start capturing

            image_path = "/Users/hyder/Downloads/VisionVista/Res/captured_image1.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as '{image_path}'")

            text_to_speech("Image successfully captured. Extracting text from image.")
            process_image_and_read(image_path)

            ask_for_another_image()  # Ask if the user wants another image

            command_event.clear()  # Clear the event so we stop image capture
            pause_listening_event.set()  # Resume listening for commands

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_audio():
    """
    Record audio from the microphone using sounddevice for the specified duration and save it as a .wav file.
    """
    try:
        print("Recording...")
        audio_data = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
        sd.wait()  # Wait until the recording is finished

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit format (int16)
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())
        wf.close()

        return WAVE_OUTPUT_FILENAME
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def transcribe_audio(audio_file):
    """
    Send the recorded audio file to OpenAI's Whisper API for transcription.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {
        'file': open(audio_file, 'rb'),
        'model': (None, 'whisper-1'),
    }

    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()  # Raises an exception for bad responses (4xx, 5xx)

        return response.json().get('text', '')
    except requests.exceptions.RequestException as e:
        print(f"Error during transcription: {e}")
        return None

def listen_for_command(command_event):
    while True:
        pause_listening_event.wait()  # This ensures the listening is paused until needed

        audio_file = record_audio()
        if audio_file:
            print("Transcribing audio...")
            user_command = transcribe_audio(audio_file)

            if user_command:
                print(f"User said: {user_command}")

                if "capture" in user_command.lower():
                    command_event.set()  # Set the event to trigger image capture

                elif "exit" in user_command.lower():
                    text_to_speech("Ok, exiting the reading mode.")
                    os._exit(0)

                elif "yes" in user_command.lower():
                    command_event.set()  # Set the event again to capture another image

                elif "no" in user_command.lower():
                    text_to_speech("Ok, exiting the reading mode.")
                    os._exit(0)
        else:
            print("Audio recording failed.")

def process_image_and_read(image_path):
    print(f"Processing the image: {image_path}")
    
    description = get_openai_response(image_path)
    
    if description:
        print(f"Extracted Text: {description}")
        text_to_speech(description)
    else:
        print("No text found in the image.")
        text_to_speech("No text found in the image.")

def ask_for_another_image():
    text_to_speech("Do you want to capture another image? Please say yes or no.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_response(image_path, user_query="Extract text from this image"):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    content = [
        {"type": "text", "text": user_query},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]

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
    speech_file_path = '/Users/hyder/Downloads/VisionVista/Res/response_text1.mp3'

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
    text_to_speech("Welcome to the reading mode. Say capture to capture the image or exit to close the reading mode.")

if __name__ == "__main__":
    welcome_message()

    command_event = threading.Event()

    voice_thread = threading.Thread(target=listen_for_command, args=(command_event,))
    voice_thread.start()

    pause_listening_event.set()

    capture_image_from_camera(command_event)
