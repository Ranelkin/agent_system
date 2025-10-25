from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
import re
import json

logger = setup_logging("comment_agent")

class CommentState(TypedDict):
    messages: Annotated[list, add_messages]
    documentation_results: dict | None

def comment_codebase(state: CommentState) -> CommentState:
    """Graph node that calls comment_codebase tool via MCP"""
    logger.info("Comment node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract directory path
    path_match = re.search(r'(/[^\s]+)', last_message)
    dir_path = path_match.group(1) if path_match else None
    
    if not dir_path:
        response = "Please provide a valid directory path for documentation."
        return {
            "messages": [{"role": "assistant", "content": response}],
            "documentation_results": None
        }
    
    logger.info(f"Documenting codebase at: {dir_path}")
    
    # Call via your existing MCPSessionManager
    manager = get_mcp_manager()
    result_str = manager.mcp_tool.call_tool("comment_codebase", {"dir": dir_path})
    
    # Parse JSON result
    try:
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
    except:
        result = {"status": "unknown", "message": result_str}
    
    response_content = f"Documentation completed:\n"
    response_content += f"Status: {result.get('status')}\n"
    response_content += f"Files processed: {result.get('count')}\n"
    if result.get('message'):
        response_content += f"Message: {result.get('message')}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "documentation_results": result
    }

def create_comment_agent_graph() -> StateGraph:
    """Factory function for comment agent subgraph"""
    logger.info("Creating comment agent graph")
    
    builder = StateGraph(CommentState)
    builder.add_node("comment_agent", comment_codebase)
    builder.add_edge(START, "comment_agent")
    builder.add_edge("comment_agent", END)
    
    return builder.compile()