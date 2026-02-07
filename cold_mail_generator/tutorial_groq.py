import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from env_config import load_env
from langchain_groq import ChatGroq

load_env()

llm = ChatGroq(
    temperature = 0,
    groq_api_key = os.getenv("groq_api_key"),
    model_name = os.getenv("llama_model_name")
)

response = llm.invoke("The first perosn to land on moon was ...")

print(response.content)