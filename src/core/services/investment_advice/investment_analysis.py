from langgraph.graph import StateGraph, START, END
from typing import Literal
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
from ....model import llm
from .util import InvestmentState
import re

logger = setup_logging("investment_agent")


def fetch_company_data(state: InvestmentState) -> InvestmentState:
    """Fetches company data - handles both direct and subgraph invocation"""
    
    # Extract ticker from state if present, otherwise from messages
    ticker = state.get('ticker')
    
    if not ticker:
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', last_message)
        ticker = ticker_match.group(1) if ticker_match else None
    
    if not ticker:
        return {
            "messages": [{"role": "assistant", "content": "Please provide a valid stock ticker (e.g., AAPL, TSLA, GOOGL)."}],
            "ticker": ""
        }
    
    logger.info(f"Fetching data for ticker: {ticker}")
    
    manager = get_mcp_manager()
    
    # Search for fundamental data
    fundamental_query = f"{ticker} stock financial data earnings revenue P/E ratio"
    fundamental_data = manager.mcp_tool.call_tool("perform_web_search", {"search_term": fundamental_query})
    
    # Search for technical data
    technical_query = f"{ticker} stock price momentum technical analysis chart"
    technical_data = manager.mcp_tool.call_tool("perform_web_search", {"search_term": technical_query})
    
    combined_data = f"""
    === FUNDAMENTAL DATA ===
    {fundamental_data}
    
    === TECHNICAL DATA ===
    {technical_data}
    """
    
    return {
        "ticker": ticker.upper(),
        "company_data": combined_data,
        "discussion_round": 0,
        "messages": [{"role": "system", "content": f"Fetched data for {ticker}"}]
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
    logger.info(response.content)
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
    logger.info(response.content)
    return {
        "technical_analysis": response.content,
        "messages": [{"role": "assistant", "content": f"[Technical Analyst]: {response.content}"}]
    }

def extract_ticker_node(state: InvestmentState) -> InvestmentState:
    """Extract ticker from user message - supports both $TSLA and TSLA formats"""
    ticker = state.get('ticker')
    
    if not ticker:
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""
        
        ticker_match = re.search(r'\$([A-Z]{1,5})\b', last_message)
        
        if not ticker_match:
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', last_message)
        
        ticker = ticker_match.group(1).upper() if ticker_match else None
    
    if not ticker:
        return {
            "messages": [{"role": "assistant", "content": "Please provide a valid stock ticker (e.g., $TSLA, $AAPL, $GOOGL)."}],
            "ticker": ""
        }
    
    logger.info(f"Extracted ticker: {ticker}")
    
    return {
        "ticker": ticker,
        "discussion_round": 0
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
    
    SENTIMENT ANALYST'S VIEW: 
    {state.get('sentiment_analysis', 'Not available')}
    Provide a final synthesis:
    1. **Consensus Points**: Where do the analysts agree?
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
    
    # Round 0: Fundamental, technical, sentiment agent's initial analysis
    # Round 1: Fundamental, technical, sentiment agent's responds to the other agents analyses
    # Round 3: Synthecise 
    # The formula for ammount of round = 2 * agent_ammount + 1 
    if round_num == 7: 
        return "mediator"
    
    assign = round_num % 3 
    if assign == 0: 
        return "fundamental"
    elif assign == 1: 
        return "technical"
    elif assign == 2: 
        return "sentiment"
    else: 
        return "end"
    


def create_investment_agent_graph() -> StateGraph:
    logger.info("Creating investment agent graph")
    
    builder = StateGraph(InvestmentState)
    
    builder.add_node("extract_ticker", extract_ticker_node) 
    builder.add_node("fetch_data", fetch_company_data)
    builder.add_node("fundamental", fundamental_agent_node)
    builder.add_node("technical", technical_agent_node)
    builder.add_node("mediator", mediator_node)
    builder.add_node("increment", increment_round)
    
    builder.add_edge(START, "extract_ticker")  
    builder.add_edge("extract_ticker", "fetch_data")  
    builder.add_edge("fetch_data", "increment")
    
    # Conditional routing for discussion rounds
    builder.add_conditional_edges(
        "increment",
        route_discussion,
        {
            "fundamental": "fundamental",
            "technical": "technical",
            "sentiment": "sentiment",
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
    