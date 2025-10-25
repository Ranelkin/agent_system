from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
import re
import json

logger = setup_logging("test_agent")

class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    test_results: dict | None

def create_unit_tests(state: TestState) -> TestState:
    """Graph node that calls create_unit_tests tool via MCP"""
    logger.info("Test node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract directory path
    path_match = re.search(r'(/[^\s]+)', last_message)
    dir_path = path_match.group(1) if path_match else None
    
    if not dir_path:
        response = "Please provide a valid directory path for creating tests."
        return {
            "messages": [{"role": "assistant", "content": response}],
            "test_results": None
        }
    
    logger.info(f"Creating tests for: {dir_path}")
    
    manager = get_mcp_manager()
    result_str = manager.mcp_tool.call_tool("create_unit_tests", {"dir": dir_path})
    
    # Parse JSON result
    try:
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
    except:
        result = {"status": "unknown", "message": result_str}
    
    response_content = f"Test creation completed:\n"
    response_content += f"Status: {result.get('status')}\n"
    response_content += f"Files processed: {result.get('count')}\n"
    if result.get('message'):
        response_content += f"Message: {result.get('message')}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "test_results": result
    }

def create_test_agent_graph() -> StateGraph:
    """Factory function for test agent subgraph"""
    logger.info("Creating test agent graph")
    
    builder = StateGraph(TestState)
    builder.add_node("test_agent", create_unit_tests)
    builder.add_edge(START, "test_agent")
    builder.add_edge("test_agent", END)
    
    return builder.compile()