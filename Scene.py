import cv2
import threading
import time
import requests
import base64
import playsound
import os
import pyaudio
import wave

# Set your OpenAI API key directly here
api_key = "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"

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "command.wav"

# Pause listening event to control when voice listener is active
pause_listening_event = threading.Event()

# Global variable to manage modes
mode = "initial"  # Modes: 'initial', 'interactive'
current_image_path = None  # Store the path to the last captured image

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

        # Check if the capture command has been received
        if command_event.is_set() and mode == "initial":
            # Pause voice listening while capturing and processing the image
            pause_listening_event.clear()

            current_image_path = "/Users/hyder/Downloads/VisionVista/Res/captured_image.jpg"
            cv2.imwrite(current_image_path, frame)
            print(f"Image saved as '{current_image_path}'")

            text_to_speech("Image successfully captured. Let me explain what's in this image.")
            process_image_and_read(current_image_path)

            # Switch mode to interactive after explaining the image
            mode = "interactive"

            # Ask if the user has any further questions
            ask_for_further_questions()

            # Clear the command event to allow new commands
            command_event.clear()

            # Resume voice listening after processing the image
            pause_listening_event.set()

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    print("Listening for voice command...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a .wav file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

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
        # Only listen for voice commands when not paused
        pause_listening_event.wait()

        audio_file = record_audio()
        user_command = transcribe_audio(audio_file)

        if user_command:
            print(f"User said: {user_command}")

            if mode == "initial":
                # In initial mode, only respond to "capture" and "exit"
                if "capture" in user_command.lower():
                    command_event.set()  # Trigger image capture

                elif "exit" in user_command.lower():
                    text_to_speech("Ok, exiting the program.")
                    os._exit(0)

                else:
                    # If user says anything else during initial mode
                    text_to_speech("I don't understand. Please say 'capture' to take a picture or 'exit' to leave.")

            elif mode == "interactive":
                # In interactive mode, any query will be processed
                if "exit" in user_command.lower():
                    text_to_speech("Ok, exiting the program.")
                    os._exit(0)
                else:
                    # Process the user's question about the image
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
        response = get_openai_response(current_image_path, user_query)  # Send user query and image to OpenAI

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

    # Create the proper payload
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
    speech_file_path = 'response_text.mp3'

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

    # Start voice listening in a separate thread
    voice_thread = threading.Thread(target=listen_for_command, args=(command_event,))
    voice_thread.start()

    # Allow voice listening from the start
    pause_listening_event.set()

    # Capture images in the main thread to avoid OpenCV thread errors
    capture_image_from_camera(command_event)
