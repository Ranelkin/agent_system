from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
from ....model.agent import llm
import re

logger = setup_logging("investment_agent")

class InvestmentState(TypedDict):
    messages: Annotated[list, add_messages]
    ticker: str
    company_data: str | None
    fundamental_analysis: str | None
    technical_analysis: str | None
    discussion_round: int
    final_recommendation: str | None


def fetch_company_data(state: InvestmentState) -> InvestmentState:
    """Fetches company data from the web"""
    logger.info(f"Fetching data for ticker: {state['ticker']}")
    
    manager = get_mcp_manager()
    
    # Search for fundamental data
    fundamental_query = f"{state['ticker']} stock financial data earnings revenue P/E ratio"
    fundamental_data = manager.mcp_tool.call_tool("perform_web_search", {"search_term": fundamental_query})
    
    # Search for technical data
    technical_query = f"{state['ticker']} stock price momentum technical analysis chart"
    technical_data = manager.mcp_tool.call_tool("perform_web_search", {"search_term": technical_query})
    
    combined_data = f"""
    === FUNDAMENTAL DATA ===
    {fundamental_data}
    
    === TECHNICAL DATA ===
    {technical_data}
    """
    
    return {
        "company_data": combined_data,
        "messages": [{"role": "system", "content": f"Fetched data for {state['ticker']}"}]
    }


def fundamental_agent_node(state: InvestmentState) -> InvestmentState:
    """Analyzes based on company fundamentals"""
    logger.info(f"Fundamental agent analyzing {state['ticker']}...")
    
    round_num = state.get("discussion_round", 0)
    
    # First round: Initial analysis
    if round_num == 0:
        context = f"""You are a fundamental analysis expert.
        Analyze {state['ticker']} based on this data:
        
        {state.get('company_data', 'No data available')}
        
        Focus on: financials, P/E ratios, earnings growth, competitive position, management quality.
        Provide a clear BUY/HOLD/SELL assessment with reasoning."""
    
    # Second round: Response to technical agent
    else:
        context = f"""You are a fundamental analysis expert.
        
        Your previous analysis:
        {state.get('fundamental_analysis', '')}
        
        Technical analyst's perspective:
        {state.get('technical_analysis', '')}
        
        Respond to the technical analyst's points. Do you agree? Disagree? 
        Provide counterpoints or supporting arguments from a fundamental perspective."""
    
    response = llm.invoke([{"role": "user", "content": context}])
    
    return {
        "fundamental_analysis": response.content,
        "messages": [{"role": "assistant", "content": f"[Fundamental Analyst]: {response.content}"}]
    }


def technical_agent_node(state: InvestmentState) -> InvestmentState:
    """Analyzes based on momentum and technical indicators"""
    logger.info(f"Technical agent analyzing {state['ticker']}...")
    
    round_num = state.get("discussion_round", 0)
    
    # First round: Initial analysis
    if round_num == 0:
        context = f"""You are a technical/momentum analysis expert.
        Analyze {state['ticker']} based on this data:
        
        {state.get('company_data', 'No data available')}
        
        Focus on: price trends, momentum indicators, support/resistance levels, 
        volume patterns, moving averages.
        Provide a clear BUY/HOLD/SELL assessment with reasoning."""
    
    # Second round: Response to fundamental agent
    else:
        context = f"""You are a technical/momentum analysis expert.
        
        Your previous analysis:
        {state.get('technical_analysis', '')}
        
        Fundamental analyst's perspective:
        {state.get('fundamental_analysis', '')}
        
        Respond to the fundamental analyst's points. How does price action 
        support or contradict their view? Provide technical perspective."""
    
    response = llm.invoke([{"role": "user", "content": context}])
    
    return {
        "technical_analysis": response.content,
        "messages": [{"role": "assistant", "content": f"[Technical Analyst]: {response.content}"}]
    }


def mediator_node(state: InvestmentState) -> InvestmentState:
    """Synthesizes both analyses after discussion"""
    logger.info("Mediator synthesizing final recommendation...")
    
    synthesis_prompt = f"""You are a senior investment advisor mediating between two expert analysts.
    
    TICKER: {state['ticker']}
    
    FUNDAMENTAL ANALYST'S VIEW:
    {state.get('fundamental_analysis', 'Not available')}
    
    TECHNICAL ANALYST'S VIEW:
    {state.get('technical_analysis', 'Not available')}
    
    Provide a final synthesis:
    1. **Consensus Points**: Where do both analysts agree?
    2. **Disagreements**: Where do they differ and why?
    3. **Final Recommendation**: BUY/HOLD/SELL with conviction level (1-10)
    4. **Risk Assessment**: Key risks to this recommendation
    5. **Time Horizon**: Short-term vs long-term outlook
    
    Be decisive but acknowledge uncertainty."""
    
    response = llm.invoke([{"role": "user", "content": synthesis_prompt}])
    
    return {
        "final_recommendation": response.content,
        "messages": [{"role": "assistant", "content": f"[Senior Advisor - Final Recommendation]: {response.content}"}]
    }


def increment_round(state: InvestmentState) -> InvestmentState:
    """Increment discussion round counter"""
    current_round = state.get("discussion_round", 0)
    return {"discussion_round": current_round + 1}


def route_discussion(state: InvestmentState) -> Literal["fundamental", "technical", "mediator", "end"]:
    """Routes between agents for two-fold discussion"""
    round_num = state.get("discussion_round", 0)
    
    # Round 0: Fundamental agent's initial analysis
    # Round 1: Technical agent's initial analysis
    # Round 2: Fundamental agent responds to technical
    # Round 3: Technical agent responds to fundamental
    # Round 4: Mediator synthesizes
    
    if round_num == 0:
        return "fundamental"
    elif round_num == 1:
        return "technical"
    elif round_num == 2:
        return "fundamental"
    elif round_num == 3:
        return "technical"
    elif round_num == 4:
        return "mediator"
    else:
        return "end"


def create_investment_agent_graph() -> StateGraph:
    """Factory for investment advice subgraph with iterative discussion"""
    logger.info("Creating investment agent graph")
    
    builder = StateGraph(InvestmentState)
    
    # Add nodes
    builder.add_node("fetch_data", fetch_company_data)
    builder.add_node("fundamental", fundamental_agent_node)
    builder.add_node("technical", technical_agent_node)
    builder.add_node("mediator", mediator_node)
    builder.add_node("increment", increment_round)
    
    # Flow with two-fold discussion
    builder.add_edge(START, "fetch_data")
    builder.add_edge("fetch_data", "increment")
    
    # Conditional routing for discussion rounds
    builder.add_conditional_edges(
        "increment",
        route_discussion,
        {
            "fundamental": "fundamental",
            "technical": "technical",
            "mediator": "mediator",
            "end": END
        }
    )
    
    # After each agent speaks, increment and route again
    builder.add_edge("fundamental", "increment")
    builder.add_edge("technical", "increment")
    builder.add_edge("mediator", END)
    
    return builder.compile()


def analyze_investment(ticker: str) -> dict:
    """Main entry point for investment analysis"""
    logger.info(f"Starting investment analysis for: {ticker}")
    
    try:
        # Create initial state
        initial_state = {
            "messages": [{"role": "user", "content": f"Analyze {ticker}"}],
            "ticker": ticker.upper(),
            "company_data": None,
            "fundamental_analysis": None,
            "technical_analysis": None,
            "discussion_round": 0,
            "final_recommendation": None
        }
        
        # Run the graph
        graph = create_investment_agent_graph()
        result = graph.invoke(initial_state)
        
        return {
            "status": "success",
            "ticker": ticker,
            "recommendation": result.get("final_recommendation"),
            "message": f"Completed multi-agent analysis for {ticker}"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze {ticker}: {e}", exc_info=True)
        return {
            "status": "error",
            "ticker": ticker,
            "message": str(e)
        }
        
        
# Still in investment_service.py

def investment_node(state: InvestmentState) -> InvestmentState:
    """Graph node that extracts ticker and runs analysis"""
    logger.info("Investment node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract ticker symbol (e.g., "AAPL", "TSLA")
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', last_message)
    ticker = ticker_match.group(1) if ticker_match else None
    
    if not ticker:
        response = "Please provide a valid stock ticker (e.g., AAPL, TSLA, GOOGL)."
        return {
            "messages": [{"role": "assistant", "content": response}]
        }
    
    # Run the analysis
    result = analyze_investment(ticker)
    
    response_content = f"Investment Analysis for {ticker}:\n\n{result.get('recommendation', 'Analysis failed')}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}]
    }
    