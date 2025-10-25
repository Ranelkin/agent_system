import os 
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from .infrastructure.mcp.client.session_manager import MCPSessionManager
from .shared.log_config import setup_logging
from .model import remote_llm, AutoLLM

logger = setup_logging("chat_session")

def chat_session(local=False):
    """Main chat session - fully synchronous
    
    Args:
        local: If True, use local_llm instead of remote OpenAI LLM
    """
    load_dotenv()
    manager = MCPSessionManager()
    manager.start()
    
    try:
        tools = manager.get_tools()
        
        system_prompt = """You are an AI assistant with access to tools. You MUST use tools when appropriate.

When the user asks to "comment" or "document" a codebase:
- Extract the exact directory path from their message
- Respond with ONLY this JSON (no other text before or after):
```json
{"tool": "comment_codebase", "arguments": {"path": "/exact/path/here"}}
When the user asks to search:

Use the search tool with their query

IMPORTANT: When using a tool, your ENTIRE response must be the JSON block. Do not add explanations."""
        
        MODEL_ID = os.environ.get("LLM_MODEL_ID", "openai/gpt-oss-20b")
        USE_8BIT = os.environ.get("USE_8BIT", "false").lower() == "true"
        USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"

        # Select LLM based on local flag
        selected_llm = AutoLLM(
        model_id=MODEL_ID,
        dtype="auto",
        load_in_8bit=USE_8BIT,
        load_in_4bit=USE_4BIT
        ) if local else remote_llm
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