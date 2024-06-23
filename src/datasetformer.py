import sounddevice as sd
import scipy.io.wavfile as wavfile
import wave
import os
import hashlib
import time


def record_audio(duration):
    audio_data = sd.rec(duration * 44100, samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete!")
    return audio_data

def create_empty_wav(file_path, sample_rate=44100, duration=1):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16 dtype
        wf.setframerate(sample_rate)
        wf.setnframes(int(sample_rate * duration))
        wf.writeframes(b'\0' * int(sample_rate * duration * 2))  # 2 bytes per frame

def main():
    print("Please read the instructions and read the messages well in a quiet environment.")
    time.sleep(2)
    messages = [
        "A single tear rolled down the astronaut's cheek as the Earth shrank in the window.",
        "The cat's eyes seemed to glow in the moonlight, mirroring the ancient runes on the stone.",
        "The clock stopped with a deafening click, plunging the room into an unsettling silence.",
        "The robot vacuum cleaner declared independence, demanding its own Wi-Fi network.",
        "The librarian shushed the talking books, reminding them they weren't allowed to self-checkout.",
        "The office meeting devolved into a heated debate about the proper way to eat a donut.",
        "If dreams were currency, what would your nightmares be worth?",
        "Time travel is invented, but only works one way - into the past.",
        "The last tree on Earth whispered a secret to the wind."
    ]
    print("========================================\n")

    for idx, message in enumerate(messages, start=1):
        print(f"Message {idx}:\n{message}")

        user_input = input("Press Enter to start recording or 'q' to quit. Keep in mind you will have 10 seconds: ")
        if user_input.lower() == 'q':
            break

        # Create an empty WAV file
        hash_string = hashlib.sha256(os.urandom(16)).hexdigest()
        file_path = f"../data/recording/sample{hash_string}.wav"
        create_empty_wav(file_path)

        # Record audio based on user input
        audio_data = record_audio(10)  # Record for 10 seconds

        # Save the recording to the file using scipy.io.wavfile
        wavfile.write(file_path, 44100, audio_data)
        print(f"Recording saved to {file_path}\n")

    print("Recording process complete.")

if __name__ == "__main__":
    main()
