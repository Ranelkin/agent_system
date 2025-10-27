# CA: This file defines the main orchestrator graph. It wires a router node to dispatch to available agent subgraphs using an LLM for routing decisions. Comments added to improve readability and maintain maintainability without changing functional behavior.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ...model import llm
from ...shared.log_config import setup_logging
from ...core.services.grouped_tools import get_available_graphs, get_graph_names

logger = setup_logging("main_graph")

class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str | None

def router_node(state: MainState) -> MainState:
    """
    Routes to appropriate agent based on user message
    Uses LLM to determine which agent to use
    """
    # CA: Router node determines the target agent by consulting an LLM with the latest user message.
    logger.info("Router node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Get available agent names
    agent_names = get_graph_names()
    
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

def route_to_agent(state: MainState) -> str:
    """Conditional edge function to route to the correct agent"""
    # CA: Decide the next hop in the graph from the router output. Falls back to a default agent if absent.
    next_agent = state.get("next_agent", "search_agent")
    logger.info(f"Routing decision: {next_agent}")
    return next_agent

def create_main_graph() -> StateGraph:
    """Assembles all agent subgraphs into main orchestrator graph"""
    # CA: Build the main graph by integrating all available agent subgraphs dynamically.
    logger.info("Creating main orchestrator graph")
    
    builder = StateGraph(MainState)
    
    # Add router node
    builder.add_node("router", router_node)
    
    # Automatically register all agent graphs as subgraphs
    agent_factories = get_available_graphs()
    agent_names = get_graph_names()
    
    for factory, name in zip(agent_factories, agent_names):
        graph = factory()
        builder.add_node(name, graph)
        logger.info(f"Registered agent: {name}")
        # CA: Each agent graph is registered under its corresponding name for dynamic routing.
    
    # Add edges
    builder.add_edge(START, "router")
    
    # Conditional routing from router to agents
    builder.add_conditional_edges(
        "router",
        route_to_agent,
        {name: name for name in agent_names}
    )
    
    # All agents route to END
    for name in agent_names:
        builder.add_edge(name, END)
    
    compiled = builder.compile()
    logger.info("Main graph compiled successfully")
    
    return compiled

# Create the main graph instance
main_graph = create_main_graph()