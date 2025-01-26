# pip install google-generativeai

import google.generativeai as genai

genai.configure(api_key = "YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    prompt,
    generation_config={
        "max_output_tokens":6000,
        "temperature":0.2 
    })

