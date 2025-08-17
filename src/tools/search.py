from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv()

web_search = TavilySearch(max_results=2)

def search_web(search: str): 
    return web_search.invoke(search)


if __name__ == '__main__': 
    out = web_search.invoke("What's a 'node' in LangGraph?")
    print(out)