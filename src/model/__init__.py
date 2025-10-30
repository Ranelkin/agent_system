from .agent import llm as remote_llm 
from .local_llm import  AutoLLM
import os 

def set_llm(): 
    
    local = os.environ.get("LOCAL")
    if local=='True': 
        llm = AutoLLM()
    else: 
        llm = remote_llm
        
    return llm 

llm = set_llm()


