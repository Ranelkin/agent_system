from typing import List, Dict, Any

# Registry of MCP tools with metadata for LangChain Tool creation
TOOL_REGISTRY: List[Dict[str, Any]] = [
    {
        "mcp_name": "search",
        "langchain_name": "web_search",
        "description": "Search the web for information",
        "arg_mapping": lambda args: {"search_term": args[0]},  # Map first LangChain arg to MCP's search_term
        "arg_type": str,  # Expected input type for LangChain
    },
    {
        "mcp_name": "document_codebase",
        "langchain_name": "document_codebase",
        "description": "Generate documentation for a codebase. Pass the full directory path as the argument.",
        "arg_mapping": lambda args: {"dir": args[0]},  # Map first LangChain arg to MCP's dir
        "arg_type": str,
    },
    # Add new tools here, e.g.:
    # {
    #     "mcp_name": "new_tool",
    #     "langchain_name": "new_tool",
    #     "description": "Does something with the input string.",
    #     "arg_mapping": lambda args: {"input": args[0]},
    #     "arg_type": str,
    # }
]

def get_tool_configs() -> List[Dict[str, Any]]:
    """Returns the tool registry for LangChain Tool creation."""
    return TOOL_REGISTRY