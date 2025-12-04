
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class InvestmentState(TypedDict): 
    messages: Annotated[list, add_messages]
    ticker: str 
    company_data: str | None 
    fundamental_analysis: str | None 
    technical_analysis: str | None 
    sentiment_analysis: str | None 
    discussion_round: int 
    sentiment_data: str
    final_recommendation: str | None

