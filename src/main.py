import argparse
import os 
from .shared.log_config import setup_logging
from dotenv import load_dotenv
from .chat_session import chat_session


load_dotenv()

logger = setup_logging("main")

def main():
    
    LOCAL = False 
    parser = argparse.ArgumentParser(description="Process some files with options.")
    parser.add_argument("--local", action="store_true",  help="Flag for local inference")
    logger.info("Starting chat session.")
    args = parser.parse_args()
    if args.local: 
        LOCAL = True 
    
    os.environ['LOCAL'] = True 
    
    chat_session(LOCAL)


if __name__ == '__main__': 
    main()