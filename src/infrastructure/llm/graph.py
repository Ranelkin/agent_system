from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from ...model import llm
from ...shared.log_config import setup_logging

from ...core.services.search.search_service import search_node
from ...core.services.test.test_service import create_unit_tests
from ...core.services.documentation.comment_code_service import comment_codebase_node
from ...core.services.investment_advice.sentiment_analysis import sentiment_agent_node 
from ...core.services.investment_advice.investment_analysis import (
    extract_ticker_node,
    fetch_company_data,
    fundamental_agent_node,
    technical_agent_node,
    mediator_node,
    increment_round,
    route_discussion
)

logger = setup_logging("main_graph")

class MainState(TypedDict):
    """Main state that supports all agent types"""
    messages: Annotated[list, add_messages]
    next_agent: str | None
    ticker: str | None
    company_data: str | None
    fundamental_analysis: str | None
    technical_analysis: str | None
    discussion_round: int | None
    final_recommendation: str | None
    test_results: dict | None
    documentation_results: dict | None
    search_results: str | None


def router_node(state: MainState) -> MainState:
    """
    Routes to appropriate agent based on user message
    Uses LLM to determine which agent to use
    """
    # CA: Router node determines the target agent by consulting an LLM with the latest user message.
    logger.info("Router node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Use LLM to determine routing
    routing_prompt = f"""You must respond with ONLY ONE WORD - the agent name, nothing else.

Available agents:
- search_agent: For web searches
- test_agent: For creating unit tests
- comment_agent: For documenting code
- investment_agent: For stock analysis and investment advice

User message: {last_message}

Respond with ONLY the agent name (one word):"""
    
    routing_messages = [{"role": "user", "content": routing_prompt}]
    response = llm.invoke(routing_messages)
    
    next_agent = response.content.strip().lower()
    logger.info(f"Routing to agent: {next_agent}")
    
    return {"next_agent": next_agent}


def route_to_agent(state: MainState) -> Literal["search_agent", "test_agent", "comment_agent", "investment_agent"]:
    """Conditional edge function to route to the correct agent"""
    # CA: Decide the next hop in the graph from the router output. Falls back to search agent if absent.
    next_agent = state.get("next_agent", "search_agent")
    logger.info(f"Routing decision: {next_agent}")
    
    # Map to the actual starting node for each agent
    agent_start_map = {
        "search_agent": "search_agent",
        "test_agent": "test_agent",
        "comment_agent": "comment_agent",
        "investment_agent": "investment_agent"
    }
    
    return agent_start_map.get(next_agent, "search_agent")


def route_investment_steps(state: MainState) -> Literal["fundamental", "technical", "sentiment", "mediator", "end"]:
    """Routes between investment agent steps for multi-round discussion"""
    round_num = state.get("discussion_round", 0)
    
    # Round 0: fundamental, 1: technical, 2: sentiment
    # Round 3: fundamental, 4: technical, 5: sentiment
    # Round 6: mediator
    
    if round_num == 6:
        return "mediator"
    elif round_num > 6:
        return "end"
    
    assign = round_num % 3
    if assign == 0:
        return "fundamental"
    elif assign == 1:
        return "technical"
    else:
        return "sentiment"


def create_main_graph() -> StateGraph:
    """Assembles all agent nodes into a single flattened main orchestrator graph"""
    # CA: Build the main graph by integrating all agent nodes directly for streaming visibility.
    logger.info("Creating flattened main orchestrator graph")
    
    builder = StateGraph(MainState)
    
    builder.add_node("router", router_node)
    
    builder.add_node("search_agent", search_node)
    
    def test_agent_wrapper(state: MainState) -> MainState:
        """Wrapper to call create_unit_tests with MainState"""
        from ...core.services.test.test_service import create_unit_tests as test_func
        import re
        import json
        from ...infrastructure.mcp.client import get_mcp_manager
        
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
    
    builder.add_node("test_agent", test_agent_wrapper)
    
    # ========== COMMENT AGENT ==========
    builder.add_node("comment_agent", comment_codebase_node)
    
    # ========== INVESTMENT AGENT (all nodes flattened) ==========
    builder.add_node("investment_agent", extract_ticker_node)  # Entry point
    builder.add_node("fetch_data", fetch_company_data)
    builder.add_node("fundamental", fundamental_agent_node)
    builder.add_node("technical", technical_agent_node)
    builder.add_node("mediator", mediator_node)
    builder.add_node("increment", increment_round)
    builder.add_node("sentiment", sentiment_agent_node)
    # ========== EDGES: START & ROUTER ==========
    builder.add_edge(START, "router")
    
    # Conditional routing from router to agents
    builder.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "search_agent": "search_agent",
            "test_agent": "test_agent",
            "comment_agent": "comment_agent",
            "investment_agent": "investment_agent"  # Routes to extract_ticker
        }
    )
    
    # ========== EDGES: SIMPLE AGENTS → END ==========
    builder.add_edge("search_agent", END)
    builder.add_edge("test_agent", END)
    builder.add_edge("comment_agent", END)
    
    # ========== EDGES: INVESTMENT AGENT FLOW ==========
    # Investment agent entry → fetch data
    builder.add_edge("investment_agent", "fetch_data")
    
    # Fetch data → increment (to start discussion)
    builder.add_edge("fetch_data", "increment")
    
    # Conditional routing for discussion rounds
    builder.add_conditional_edges(
        "increment",
        route_investment_steps,
        {
            "fundamental": "fundamental",
            "technical": "technical",
            "sentiment": "sentiment",
            "mediator": "mediator",
            "end": END
        }
    )
    
    # After each analyst speaks, increment round and route again
    builder.add_edge("fundamental", "increment")
    builder.add_edge("technical", "increment")
    builder.add_edge("sentiment", "increment")
    # Mediator produces final recommendation → END
    builder.add_edge("mediator", END)
    
    compiled = builder.compile()
    logger.info("Main graph compiled successfully with all nodes flattened")
    
    return compiled


# Create the main graph instance
main_graph = create_main_graph()