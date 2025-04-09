import sys

import ollama
import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

def app_session_init():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage("Hello, How can I help you?")]

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = get_models()[0]

    chat_history = st.session_state["chat_history"]

    for history in chat_history:
        if isinstance(history, AIMessage):
            st.chat_message("ai").write(history.content)
        if isinstance(history, HumanMessage):
            st.chat_message("human").write(history.content)

def get_ollama_models():
    ollama_client = Ollama()
    model_list = ollama_client.list_models()
    for model in model_list: print(f"Model Name: {model.name}, Version: {model.version}, Description: {model.description}")


def get_models():
    models = ollama.list()
    if not models:
        print("No model found, please visit: https://ollama.dev/models")
        sys.exit(1)
    models_list = []
    for model in models["models"]:
        models_list.append(model["model"])
    
    return models_list

def stream_text(text):
    for word in text.split():
        yield word + " "

def run():
    st.set_page_config(page_title="Chat Application")
    st.header("Chat :blue[Application]")

    st.selectbox("Select LLM model", get_models(), key="selected_model")

    app_session_init()
    prompt = st.chat_input("Add your prompt...")

    selected_model = st.session_state["selected_model"]
    print(f"selected_model : {selected_model}")

    llm = ChatOllama(model=selected_model, temperature=0.7)

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"] += [HumanMessage(prompt)]
        output = llm.invoke(prompt)

        with st.chat_message("ai"):
            ai_message = st.write_stream(stream_text(output.content))
        
        st.session_state["chat_history"] += [AIMessage(ai_message)]

if __name__ == "__main__":
    run()
    # models = get_models()
    # print(models)