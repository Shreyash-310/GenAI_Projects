import whisper

# Load Whisper model
model = whisper.load_model("base")

# Path to your audio file
audio_file = "EarningsCall.wav"

# Perform transcription
result = model.transcribe(audio_file)

# Get and print the transcription text
print("Transcription:", result['text'])