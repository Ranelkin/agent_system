from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ....model.agent import llm
from ....core.services.search.search_service import search
from ....shared.log_config import setup_logging

logger = setup_logging("search_agent")

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: str | None

def search_agent_node(state: SearchState):
    """
    Processes search requests by extracting search terms from messages
    and calling the search service
    """
    logger.info("Search agent node invoked")
    
    # Get the last user message
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract search term (you could use LLM to parse this more intelligently)
    # For now, simple extraction
    search_term = last_message.replace("search for", "").replace("search", "").strip()
    
    logger.info(f"Extracted search term: {search_term}")
    
    # Call the search service
    results = search(search_term)
    
    # Format response
    response_content = f"Search results for '{search_term}':\n{results}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "search_results": results
    }


def create_search_agent_graph() -> StateGraph:
    """Factory function to create the search agent subgraph"""
    logger.info("Creating search agent graph")
    
    builder = StateGraph(SearchState)
    builder.add_node("search_agent", search_agent_node)
    builder.add_edge(START, "search_agent")
    builder.add_edge("search_agent", END)
    
    return builder.compile()