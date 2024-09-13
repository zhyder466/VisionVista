import pyaudio
import wave
import numpy as np
import subprocess
from openai import OpenAI
import threading
import playsound
import warnings
import time 

client = OpenAI(api_key='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')
warnings.filterwarnings("ignore")

def readText(text):
    speech_file_path = '/Users/hyder/Downloads/VisionVista/Res/extracted_text.mp3'
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    response.stream_to_file(speech_file_path)
    playsound.playsound(speech_file_path)

def record_audio(file_name, silence_threshold=3000, silence_duration=1, sample_rate=44100, chunk_size=1024, format=pyaudio.paInt16, channels=1):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    frames = []
    silent_duration = 0

    print("Recording...")
    while True:
        data = stream.read(chunk_size)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_data))
        
        if amplitude < silence_threshold:
            silent_duration += chunk_size / sample_rate
        else:
            silent_duration = 0
        
        print("Silent Duration:", silent_duration)

        if silent_duration > silence_duration:
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    file_path = f"/Users/hyder/Downloads/VisionVista/Res/{file_name}"

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print("Audio saved to:", file_path)
    return file_path

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    
    transcription_text = transcription.text
    print(transcription_text)
    
    if "activate free walk mode" in transcription_text.lower():
        threading.Thread(target=readText, args=("Activating free walk mode",)).start()
        subprocess.run(["python", "ObjectAndFaceWithDistance.py"])
    elif "activate reading mode" in transcription_text.lower():
        threading.Thread(target=readText, args=("Activating reading mode",)).start()
        subprocess.run(["python", "Camera.py"])
    elif "activate scene description mode" in transcription_text.lower():  # New mode: Scene mode
        threading.Thread(target=readText, args=("Activating scene description mode",)).start()
        subprocess.run(["python", "scene.py"])
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
    while True:
        
        audio_file_path = record_audio("recorded_audio.wav")

        if transcribe_audio(audio_file_path):
            break

        readText("Welcome back to Vision Vista main menu, how can I help you today?")
        
if __name__ == "__main__":
    main()
