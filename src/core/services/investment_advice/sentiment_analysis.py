from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
from ....model import llm
from .util import InvestmentState

logger = setup_logging('Sentiment Agent')

def fetch_sentiment_data(state: InvestmentState) -> InvestmentState:
    """Fetches company & market sentimet data - handles only direct graph 
    invocation
    """
    ticker = state.get('ticker', 'the market')
    # If no ticker is given the task is to analyze market sentiment 
 
    logger.info('Analyzing current market sentiment')
    mcp = get_mcp_manager()
    sentiment_query = f"{ticker} investment sentiment analysis"
    #####################
    # Place holder for market sentiment fetching 
    # PLACEHOLDER 
    sentiment_data = mcp.mcp_tool.call_tool("perform_web_search", {"search_term": sentiment_query})
    #####################
    InvestmentState['sentiment_data'] = sentiment_data
    
    logger.info(f'Current market sentiment: {sentiment_data}')
 
        
    return {
        "ticker": ticker.upper() if ticker else ''
        "sentiment": sentiment_data
        "discussion_round": InvestmentState.discussion_round
        "messages" [{"role": "sentiment_analyst", 
                     "content": f"Fetched sentiment data for {ticker if ticker else 'Market'}" }]
    }

def sentiment_agent_node(state: InvestmentState) -> InvestmentState:
    """_summary_"""
    logger.info(f"Fundamental agent analyzing {state['ticker'] if state['ticker'] else 'Market sentiment'}")
    
    round_num = state.get("discussion_round", 0)
    
    # First round only provide market / stock sentiment 
    if round_num == 0: 
        context = f"""You are a market sentiment expert. 
        Analyze {state['ticker'] if state['ticker'] else 'Market sentiment'} sentiment 
        based on this data: 
        
        {state.get('sentiment_data')}
        
        Focus on buy / sell volume. Insider Trading, Analyst expectations and Volatility
        """
    # Second round: Response to technical agent
    else: 
        context = f"""You are a market sentiment expert. 
        Analyze {state['ticker'] if state['ticker'] else 'Market sentiment'} sentiment 
        based on this data: 
        
        Your previous analysis: 
        {state.get('sentiment_analysis', '')}
        
        technical analyst: 
        {state.get('technical_analysis', '')}
        
        fundamental analyst's perspective: 
        {state.get('fundamental_analysis', '')}
        Respond to the technical analyst' and fundamental analyst's perspective points. 
        How does sentiment action support or contradict their view? Provide sentiment perspective.
        """
    response = llm.invoke([{"role": "user", "content": context}])
    logger.info(response.content)
    return {
        "sentiment_analysis": response.content
        "messages": [{"role": "sentiment_analyst", "content": f"[Sentiment Analyst]: {response.content}"}]
    }