import pyaudio
import wave
import os
import threading
import time
from gtts import gTTS
import pygame
from faster_whisper import WhisperModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threading import Event

import speech_recognition as sr

stop_words = ["hi","stop", "end", "cancel", "quit", "exit","enough","that's all"]

pause_event = Event()
stop_event = Event()

NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

voice_responses = {}

def transcribe_chunk(model, file_path):
    segments, _ = model.transcribe(file_path, beam_size=1)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def record_chunk(p, stream, file_path, chunk_length=3):
    frames = []
    total_frames = int(16000 / 1024 * chunk_length)
    try:
        for _ in range(total_frames):
            if pause_event.is_set() or stop_event.is_set():
                break
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
    except IOError as e:
        print("IOError: ", e)
        time.sleep(0.01)
    finally:
        if frames:
            wf = wave.open(file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
            wf.close()

def transcribe_and_respond(model, chunk_file):
    transcription = transcribe_chunk(model, chunk_file)
    print(NEON_GREEN + transcription + RESET_COLOR)
    return transcription

def text_to_speech_with_gtts(text, file_name="response.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(file_name)
def speak_prompt(text="Now, you can ask questions..."):
    tts = gTTS(text=text, lang='en')
    prompt_file_name = "prompt.mp3"
    tts.save(prompt_file_name)
    play_audio_response(prompt_file_name)
    os.remove(prompt_file_name)

def play_audio_response(file_name="response.mp3"):
    global audio_playback_completed
    pygame.mixer.init()
    sound = pygame.mixer.Sound(file_name)
    channel = sound.play()
    while channel.get_busy():
        if pause_event.is_set() or stop_event.is_set():
            channel.stop()
            break
        time.sleep(0.1)
    pygame.mixer.quit()
    audio_playback_completed = True


def record_and_transcribe(p, stream, model, tokenizer, gpt_model):
    chunk_file = "temp_chunk.wav"
    record_chunk(p, stream, chunk_file)
    transcription = transcribe_and_respond(model, chunk_file)
    os.remove(chunk_file)

    inputs = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = gpt_model.generate(
        inputs["input_ids"], 
        max_length=50, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2, 
        temperature=0.7, 
        top_k=50
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("GPT2 Response:", response_text)
    
    global voice_responses
    voice_responses['latest'] = {'transcription': transcription, 'gpt2_response': response_text}
    print("voice_responses is",voice_responses)
    text_to_speech_with_gtts(response_text)
    play_audio_response()
    if audio_playback_completed:
        print("audio_playback_complete")
        reset_events()
        start_listening() 

    return transcription, response_text

def start_voice_interaction():
    global pause_event, stop_event

    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    tokenizer = GPT2Tokenizer.from_pretrained("C:/Users/shengjie zhao/Desktop/UI1/model/GPT2_model_new")
    
    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    gpt_model = GPT2LMHeadModel.from_pretrained("C:/Users/shengjie zhao/Desktop/UI1/model/GPT2_model_new")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    print("Start speaking...")
    
    pause_event.clear()

    speak_prompt("Now, you can ask questions...")
    responses = {
        'transcription': None,
        'gpt2_response': None,
        'is_code_response': False,
    }
    
    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                print("pause event is set") 
                break
            transcription, response_text = record_and_transcribe(p, stream, model, tokenizer, gpt_model)
            if transcription:
                print('Transcription:', transcription)
                return {'transcription': transcription, 'gpt2_response': response_text}
    except KeyboardInterrupt:
        print("Manually interrupted.")
    finally:
        pause_event.set()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Terminated.")

def stop_voice():
    stop_event.set()
    print("Stop event has been set.")
def reset_events():
    global pause_event, stop_event
    pause_event.clear()
    stop_event.clear()
    print("Events have been reset.")

def start_listening():
    reset_events()
    print("Is reset.")
    listening_thread = threading.Thread(target=listen_for_stop_command)
    listening_thread.start()
    start_voice_interaction_thread = threading.Thread(target=start_voice_interaction)
    start_voice_interaction_thread.start()

def listen_for_stop_command():
    global pause_event, stop_event
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while not stop_event.is_set():
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                text = recognizer.recognize_google(audio).lower()
                
                if any(stop_word in text for stop_word in stop_words):
                    print("'Stop' command detected. Pausing...")
                    pause_event.set()
#                    if not stop_event.is_set():
#                        threading.Timer(1, start_listening).start()
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    print("Program started. Say 'stop' to restart listening.")
    start_listening()

if __name__ == "__main__":
    main()