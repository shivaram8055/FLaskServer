import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Initialize Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to transcribe audio
def transcribe_audio(audio_data):
    input_values = processor(audio_data, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# Main function
def main():
    CHUNK = 1024
    sample_rate = 16000
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                        input=True, frames_per_buffer=CHUNK)

    print("Listening...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Transcribe audio
            transcription = transcribe_audio(audio_data)
            print("Transcription:", transcription)

        except KeyboardInterrupt:
            print("Stopping...")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()
