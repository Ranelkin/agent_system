from langchain_tavily import TavilySearch

from ....shared.config import AppConfig

config = AppConfig()

web_search = TavilySearch(max_results=2)

def search_web(search: str): 
    return web_search.invoke(search)
