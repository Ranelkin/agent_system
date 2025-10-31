from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
from .web_search import search_web

logger = setup_logging("search_agent")

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: str | None

def perform_web_search(search_term: str) -> str:
    """Perform a web search and return results as a string"""
    logger.info(f"Performing web search for: {search_term}")
    result = search_web(search_term)
    return result

# Graph node function - uses state
def search_node(state: SearchState) -> SearchState:
    """Graph node that calls search tool via MCP"""
    logger.info("Search node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract search query
    search_term = last_message.replace("search for", "").replace("search", "").strip()
    logger.info(f"Extracted search term: {search_term}")
    
    manager = get_mcp_manager()
    results = manager.mcp_tool.call_tool("perform_web_search", {"search_term": search_term})
    
    response_content = f"Search results for '{search_term}':\n{results}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "search_results": results
    }

def create_search_agent_graph() -> StateGraph:
    """Factory function for search agent subgraph"""
    logger.info("Creating search agent graph")
    
    builder = StateGraph(SearchState)
    builder.add_node("search_agent", search_node)
    builder.add_edge(START, "search_agent")
    builder.add_edge("search_agent", END)
    
    return builder.compile()