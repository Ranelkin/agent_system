from langgraph.prebuilt import create_react_agent
from .infrastructure.mcp.client.session_manager import MCPSessionManager
from .shared.log_config import setup_logging
from .model.agent import llm as remote_llm
from .model.local_llm import local_llm

logger = setup_logging("chat_session")

def chat_session(local=False):
    """Main chat session - fully synchronous
    
    Args:
        local: If True, use local_llm instead of remote OpenAI LLM
    """
    manager = MCPSessionManager()
    manager.start()
    
    try:
        tools = manager.get_tools()
        
        system_prompt = """You are an AI assistant that can search the web or document codebases.
        
When asked to document a codebase:
1. Extract the EXACT directory path from the user's message
2. Call the document_codebase tool with that exact path as the argument
3. Return the tool's output
4. Do NOT make multiple calls or suggest alternative paths

For other queries, use the Web Search tool if applicable."""
        
        # Select LLM based on local flag
        selected_llm = local_llm if local else remote_llm
        llm_type = "Local" if local else "Remote (OpenAI)"
        logger.info(f"Using {llm_type} LLM")
        
        # Create the agent with selected LLM
        # Both local_llm and remote_llm now support the same interface
        agent = create_react_agent(
            selected_llm,
            tools,
            prompt=system_prompt
        )
        
        # Chat loop
        logger.info(f"\n🤖 AI Assistant ready with {llm_type}! (type 'exit' to quit)\n")
        
        while True:
            query = input("You: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            logger.info("\nAssistant: ")
            
            try:
                response = agent.invoke(
                    {"messages": [("human", query)]},
                    config={"recursion_limit": 50}
                )
                
                # Extract and log the response message content
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1]
                    if hasattr(final_message, 'content'):
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
        logger.info("Finished processing request")
        manager.stop()
        logger.info("\n👋 Goodbye!")