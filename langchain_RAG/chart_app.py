import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from env_config import load_env
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_env()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, image[0],prompt])
    return response.text

def input_iamge_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
if __name__ == "__main__":

    st.set_page_config(page_title="Gemini Chart Demo")
    st.header("Chart Analysis Bot")

    input = st.text_input("Input Prompt: ", key="input")
    uploaded_file = st.file_uploader("Choose an image ...", type=["jpg","jpeg","png"])
    image = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    submit = st.button("Tell em about the chart")

    prompt = """
                You are ana expert in understanding charts.
                You will receive input images as charts and you will have to summarize the charts in details.
            """
    
    if submit:
        image_data = input_iamge_setup(uploaded_file)
        response = get_gemini_response(prompt, image_data, input)
        st.subheader("The response is ")
        st.write(response)