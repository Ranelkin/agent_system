import logging, asyncio
from dotenv import load_dotenv
from mcp_client.session_manager import chat_session

load_dotenv()

logger = logging.getLogger("main")


def main(): 
    logger.info("Starting chat session.")
    asyncio.run(chat_session())

if __name__ == '__main__': 
    main()
