from dotenv import load_dotenv
from .infrastructure.llm.graph import main_graph
from .infrastructure.mcp.client import get_mcp_manager, shutdown_mcp
from .shared.log_config import setup_logging

logger = setup_logging("chat_session")

def chat_session(local=False):
    """Main chat session with MCP + LangGraph integration"""
    load_dotenv()
    
    manager = get_mcp_manager()
    
    try:
        logger.info("\n🤖 AI Assistant ready!")
        logger.info("Available commands:")
        logger.info("  - 'search for X' - Web search")
        logger.info("  - 'create tests for /path' - Generate unit tests")
        logger.info("  - 'document /path' - Add documentation")
        logger.info("  - 'exit' to quit\n")
        
        while True:
            query = input("You: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            logger.info("\nAssistant: ")
            
            try:
                response = main_graph.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"recursion_limit": 50}
                )
                
                # Extract response
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1]
                    if isinstance(final_message, dict):
                        print(final_message.get("content", final_message))
                    elif hasattr(final_message, 'content'):
                        print(final_message.content)
                    else:
                        print(final_message)
                else:
                    print(response)
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}", exc_info=True)
                print(f"Error: {e}")
            
            print("\n")
            
    finally:
        shutdown_mcp()
        logger.info("\n👋 Goodbye!")