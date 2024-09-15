import pyaudio
import wave
import requests
import time
import subprocess
import threading
import warnings
from openai import OpenAI
import playsound

client = OpenAI(api_key='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')
warnings.filterwarnings("ignore")

def readText(text):
    speech_file_path = 'extracted_text.mp3'
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    response.stream_to_file(speech_file_path)
    playsound.playsound(speech_file_path)

# Audio recording parameters
CHUNK = 2048  # Increased buffer size to prevent overflow
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 16000  # Audio sample rate
RECORD_SECONDS = 5  # Duration of audio capture in seconds for each request
WAVE_OUTPUT_FILENAME = "output.wav"  # Temporary file for audio

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start audio stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

def record_audio():
    frames = []
    
    try:
        # Record audio for the specified duration
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)  # Handle buffer overflow
            frames.append(data)
    except IOError as e:
        print(f"Error: {e}")

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
        "Authorization": f"Bearer {client.api_key}"
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

def process_transcription(transcription_text):
    if "activate free walk mode" in transcription_text.lower():
        threading.Thread(target=readText, args=("Activating free walk mode",)).start()
        subprocess.run(["python", "ObjectAndFaceWithDistance.py"])
    elif "activate reading mode" in transcription_text.lower():
        threading.Thread(target=readText, args=("Activating reading mode",)).start()
        time.sleep(2)
        subprocess.run(["python", "Camera.py"])
    elif "activate scene describe mode" in transcription_text.lower():
        threading.Thread(target=readText, args=("Activating scene describe mode",)).start()
        time.sleep(2)
        subprocess.run(["python", "Scene.py"])
    elif "exit" in transcription_text.lower():
        print("Exiting program.")
        threading.Thread(target=readText, args=("Exiting the program",)).start()
        return True
    else:
        threading.Thread(target=readText, args=("Command not recognized",)).start()
        time.sleep(2)
    return False

def main():
    readText("Welcome to VisionVista main menu, how can I help you today?")
    try:
        while True:
            audio_file = record_audio()
            print("Transcribing audio...")
            transcript = transcribe_audio(audio_file)
            if transcript:
                print(f"Transcribed Text: {transcript}")
                if process_transcription(transcript):
                    break
            readText("Welcome back to Vision Vista main menu, how can I help you today?")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
