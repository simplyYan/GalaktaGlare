import json
import pyttsx3
import time
import speech_recognition as sr

def text_to_speech(text, language='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speed as needed
    engine.setProperty('voice', f'{language}+whisper')  # Choose the voice for the language
    engine.say(text)
    engine.runAndWait()

def recognize_speech(seconds):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source, timeout=seconds)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Unable to understand the audio."
    except sr.RequestError as e:
        return "Error making request to the speech recognition service; {0}".format(e)

def main():
    # Read the JSON file
    try:
        with open('ggtts.json', 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        print("File 'ggtts.json' not found in the current directory.")
        return
    except json.JSONDecodeError:
        print("Error decoding the JSON file.")
        return

    # Extract configurations
    lang = config.get('lang', 'en')
    text = config.get('text', '')
    reco_config = config.get('reco')

    if not text:
        print("Field 'text' not found in the JSON file.")
        return

    # Check if the "reco" field exists
    if reco_config:
        reco_enabled, reco_seconds = reco_config.split()
        if reco_enabled.lower() == 'true':
            reco_seconds = int(reco_seconds)
            result = recognize_speech(reco_seconds)

            # Save the result to a file named "ggtts.result"
            with open('ggtts.result', 'w') as result_file:
                result_file.write(result)
        else:
            print("Field 'reco' must have the value 'true' followed by the number of seconds.")
    else:
        # If there is no "reco" field, simply speak the text using the specified language
        text_to_speech(text, language=lang)

if __name__ == "__main__":
    main()
