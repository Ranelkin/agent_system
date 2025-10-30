
from .shared.log_config import setup_logging
from dotenv import load_dotenv
from .chat_session import chat_session

load_dotenv()

logger = setup_logging("main")

def main():
    logger.info("Starting chat session.")
    chat_session()


if __name__ == '__main__': 
    main()