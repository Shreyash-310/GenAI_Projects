import io
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from env_config import load_env
import whisper
import google.generativeai as genai

load_env() ##load all the nevironment variables

class GenerateTranscript:
    def __init__(self):
        self.model = whisper.load_model(os.getenv("WHISPER_MODEL"))
        self.audio_path = os.getenv("audio_folder")
        self.transcript_path = os.getenv("transcript_folder")

    def save_transcript(self, transcript, file_name):
        transcript_file = self.transcript_path+file_name.split(".")[0]+".txt"
        print(f"transcript_file => {transcript_file}")
        with io.open(transcript_file, 'w', encoding='utf-8') as file:
            file.write(transcript)
        print("Transcription Saved")

    def get_transcribe(self, audio_file, save_transcript=False):
        trancribe_file_path = os.path.join(self.transcript_path, audio_file[:-4]+'.txt')
        if os.path.exists(trancribe_file_path):
            with open(trancribe_file_path,'r') as file:
                transcript = file.readlines()
            transcript = ''.join(transcript)
        else:
            audio_file_path= os.path.join(self.audio_path,audio_file)
            transcript = self.model.transcribe(audio_file_path)['text']
            if save_transcript:
                self.save_transcript(transcript, audio_file)
        return transcript

class GenerateMinutes(GenerateTranscript):
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        self.save_path = os.getenv("MoM_save_folder")

    def summarize(self, transcript):
        response = self.model.generate_content(["Generate the minutes of Meeting based on the following transcript", transcript])
        self.MoM = response.text
        return response.text
    
    def save_MoM(self, MoM_content, mom_filename):
        with io.open(os.path.join(self.save_path,mom_filename), 'w', encoding='utf-8') as file:
            file.write(MoM_content)
        print(f'MoM saved for file {mom_filename}')

if __name__=='__main__':
    transcript_generator = GenerateTranscript()
    audio_file = "EarningsCall.wav" # "audio_file_1.m4a"
    transcript = transcript_generator.get_transcribe(audio_file,save_transcript=True)
    print(f"transcript \n{transcript}")
    print('*'*18)
    minutes_generator = GenerateMinutes()
    MoM = minutes_generator.summarize(transcript)
    print(f"Minutes of Meeting \n{MoM}")
    minutes_generator.save_MoM(MoM, audio_file[:-4]+'.txt')
