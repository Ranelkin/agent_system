from typing import List, Callable
from .documentation.comment_code_service import comment_codebase, document_codebase
from .search.search_service import search
from .test.test_service import create_unit_tests

TOOLS: List[Callable] = [
    comment_codebase,
    document_codebase,
    search,
    create_unit_tests
]

def get_available_tools()-> List[Callable]: 
    return TOOLS