from .agent import llm as remote_llm 
from .local_llm import  AutoLLM
import os 

def set_llm(): 
    MODEL_ID = os.environ.get("LLM_MODEL_ID", "openai/gpt-oss-20b")
    USE_8BIT = os.environ.get("USE_8BIT", "false").lower() == "true"
    USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"
    local = os.environ.get("LOCAL")
    if bool(local): 
        llm = AutoLLM(
            model_id=MODEL_ID,
            dtype="auto",
            load_in_8bit=USE_8BIT,
            load_in_4bit=USE_4BIT
        )
    else: 
        llm = remote_llm
        
    return llm 

llm = set_llm()


