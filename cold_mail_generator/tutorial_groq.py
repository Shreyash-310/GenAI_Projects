import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    temperature = 0,
    groq_api_key = os.getenv("groq_api_key"), # "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",
    model_name = os.getenv("llama_model_name")
)

response = llm.invoke("The first perosn to land on moon was ...")

print(response.content)