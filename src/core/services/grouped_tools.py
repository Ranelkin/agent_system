from typing import List, Callable
from langgraph.graph import StateGraph

from .documentation.comment_code_service import comment_codebase
from .search.search_service import search
from .test.test_service import create_unit_tests

from .search.search_service import create_search_agent_graph
from .test.test_service import create_test_agent_graph


AGENT_GRAPHS: List[Callable[[], StateGraph]] = [
    create_search_agent_graph,
    create_test_agent_graph,
]

def get_available_graphs() -> List[Callable[[], StateGraph]]:
    """Returns all registered agent graph factories"""
    return AGENT_GRAPHS


TOOLS: List[Callable] = [
    comment_codebase,
    search,
    create_unit_tests
]

def get_available_tools()-> List[Callable]: 
    return TOOLS

