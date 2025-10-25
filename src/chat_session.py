from dotenv import load_dotenv
from .infrastructure.llm.graph import main_graph
from .shared.log_config import setup_logging
from .model import remote_llm, AutoLLM

logger = setup_logging("chat_session")

def chat_session(local=False):
    """Main chat session using the orchestrated graph system"""
    load_dotenv()
    
    try:
        # Chat loop
        logger.info("\n🤖 AI Assistant ready with Graph-based agents! (type 'exit' to quit)\n")
        
        while True:
            query = input("You: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            logger.info("\nAssistant: ")
            
            try:
                # Invoke the main graph
                response = main_graph.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"recursion_limit": 50}
                )
                
                # Extract and log the response
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1]
                    if isinstance(final_message, dict):
                        logger.info(final_message.get("content", final_message))
                    elif hasattr(final_message, 'content'):
                        logger.info(final_message.content)
                    else:
                        logger.info(final_message)
                else:
                    logger.info(response)
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                logger.info(f"Error: {e}")
            
            logger.info("\n\n")
            
    finally:
        logger.info("\n👋 Goodbye!")