import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize the chat model using the specified model name
llm = init_chat_model("openai:gpt-5-nano-2025-08-07")
