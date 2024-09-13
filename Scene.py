import cv2
import base64
import requests
import playsound
import threading
import speech_recognition as sr
import warnings
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
api_key = "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"
url = 'http://172.20.10.5:8080/video'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_response(image_path, user_query="What's in this image?"):
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
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def text_to_speech(text):
    speech_file_path = '/Users/hyder/Downloads/VisionVista/Res/response_text.mp3'

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

def listen_for_query():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for further query...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_query = recognizer.recognize_google(audio)
        print(f"User said: {user_query}")
        return user_query.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def capture_image_from_camera():
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

        text_to_speech("Say 'capture' to capture the image or 'exit' to quit.")
        user_command = listen_for_query()

        if user_command == "capture":
            image_path = "/Users/hyder/Downloads/VisionVista/Res/captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as '{image_path}'")

            cap.release()
            cv2.destroyAllWindows()

            threading.Thread(target=process_image_and_interaction, args=(image_path,)).start()
            return True

        elif user_command == "exit":
            cap.release()
            cv2.destroyAllWindows()
            text_to_speech("Ok, exiting the scenario description mode.")
            sys.exit()

        else:
            text_to_speech("Sorry, I didn't understand the command.")

    return False

def process_image_and_interaction(image_path):
    text_to_speech("Image successfully captured, now providing a description of the image.")

    response = get_openai_response(image_path)

    if 'choices' in response:
        content = response['choices'][0]['message']['content']
        print(f"Extracted Content: {content}")

        text_to_speech(content)

        text_to_speech("You can ask more questions about the image, please say your question or say 'exit' to quit.")

        while True:
            user_query = listen_for_query()

            if user_query == "exit":
                text_to_speech("Ok, exiting the scenario description mode.")
                sys.exit()

            if user_query:
                response = get_openai_response(image_path, user_query)
                if 'choices' in response:
                    further_content = response['choices'][0]['message']['content']
                    print(f"Further Content: {further_content}")
                    text_to_speech(further_content)
                else:
                    text_to_speech("Sorry, I couldn't find an answer to that. Please ask again.")
            else:
                text_to_speech("Sorry, I didn't understand your question.")

if __name__ == "__main__":

    while True:
        if not capture_image_from_camera():
            break
