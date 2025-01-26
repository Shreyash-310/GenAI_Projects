import os
import whisper
import io
from pathlib import Path

model = whisper.load_model("medium")

audio_path = r"D:\GenAI-Practice\GenAI-Projects\Minutes_of_Meetings_generator"
transcript_path = r"s:/G_Drive_backup/Data_Science/GenAI/audio_transcripts/"

def get_transcription(file, save=False):
    audio_file = os.path.join(audio_path, file) #file # audio_path+file
    print(f"audio_file {audio_file}")
    input()
    if os.path.exists(audio_file):
        print("File exists, proceed with transcription")
        result = model.transcribe(audio_file)
        print("Transcription : ",result['text'])
        print('*'*8)
        if save:
            transcript_file = transcript_path+file.split(".")[0]+".txt"
            print(f"transcript_file => {transcript_file}")
            with io.open(transcript_file, 'w', encoding='utf-8') as file:
                file.write(result['text'])
            print("Transcription Saved")
    else:
        print("File not Found, check the path")

if __name__ == "__main__":
    file = "EarningsCall.wav"
    # file = "D:\GenAI-Practice\GenAI-Projects\Minutes_of_Meetings_generator\EarningsCall.wav"
    get_transcription(file)
