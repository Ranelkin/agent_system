from .web_search import search_web
from ....infrastructure.mcp.server.mcp_server import mcp 
from ....shared.log_config import setup_logging

logger = setup_logging("search_service")

@mcp.tool 
def search(search_term: str): 
    """Searches the web and returns results"""
    logger.info("Search tool invoked")
    logger.info(f"Search term: {search_term}")
    return search_web(search_term)
