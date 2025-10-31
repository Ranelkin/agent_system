import chainlit as cl
import sys
import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.infrastructure.llm.graph import main_graph
from src.infrastructure.mcp.client import get_mcp_manager, shutdown_mcp
from src.shared.log_config import setup_logging

load_dotenv()
logger = setup_logging("chainlit_app")

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    try:
        manager = get_mcp_manager()
        logger.info("MCP manager initialized")
        
        await cl.Message(
            content="🤖 **AI Assistant Ready!**\n\n"
            "- 🔍 Web searches\n"
            "- 🧪 Unit tests\n"
            "- 📝 Documentation\n"
            "- 📊 Investment analysis\n\n"
            "What would you like to do?"
        ).send()
    except Exception as e:
        logger.error(f"Error in chat start: {e}", exc_info=True)

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    
    agent_steps = {}
    
    try:
        # Process graph synchronously (LangGraph's stream is blocking)
        for event in main_graph.stream(
            {"messages": [{"role": "user", "content": message.content}]},
            stream_mode="updates",
            config={"recursion_limit": 50}
        ):
            for node_name, node_output in event.items():
                logger.info(f"Processing node: {node_name}")
                
                # Map node names to friendly names
                step_names = {
                    "router": "🔀 Routing",
                    "extract_ticker": "🎯 Extracting Ticker",
                    "fetch_data": "📡 Fetching Market Data",
                    "fundamental": "📊 Fundamental Analysis",
                    "technical": "📈 Technical Analysis",
                    "mediator": "🎓 Final Recommendation",
                    "search_agent": "🔍 Web Search",
                    "test_agent": "🧪 Creating Tests",
                    "comment_agent": "📝 Documentation",
                }
                
                step_name = step_names.get(node_name, f"⚙️ {node_name}")
                
                # Create or get step
                if node_name not in agent_steps:
                    step = cl.Step(name=step_name, type="tool")
                    await step.__aenter__()
                    agent_steps[node_name] = step
                else:
                    step = agent_steps[node_name]
                
                # Extract content from node output
                if "messages" in node_output:
                    messages = node_output["messages"]
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                        else:
                            content = getattr(msg, "content", "")
                        
                        if content and content.strip():
                            step.output = content
                            await step.send()
        
        # Close all steps
        for step in agent_steps.values():
            await step.__aexit__(None, None, None)
        
        # Send final message
        last_node = list(event.keys())[-1]
        final_output = event[last_node]
        
        if "messages" in final_output and final_output["messages"]:
            last_msg = final_output["messages"][-1]
            content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
            
            if content:
                await cl.Message(content=content).send()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        
        # Close any open steps
        for step in agent_steps.values():
            try:
                await step.__aexit__(None, None, None)
            except:
                pass
        
        await cl.Message(
            content=f"❌ **Error**: {str(e)}\n\nPlease try again."
        ).send()

@cl.on_chat_end
def end():
    """Cleanup when chat ends"""
    try:
        shutdown_mcp()
        logger.info("Session ended")
    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)