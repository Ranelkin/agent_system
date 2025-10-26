from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from ....shared.log_config import setup_logging
from ....infrastructure.mcp.client import get_mcp_manager
import re
import json
from ....util.FileBase import FileBase
from ....model import remote_llm
logger = setup_logging("comment_agent")

class CommentState(TypedDict):
    messages: Annotated[list, add_messages]
    documentation_results: dict | None



def comment_codebase(dir: str) -> dict:
    """Traverses the directory and creates documentation"""
    logger.info(f"Starting to document codebase at: {dir}")
    processed_files = []
    try:
        logger.info(f"Starting documentation for directory: {dir}")
        f_base = FileBase(dir)
        f_base.traverse()
        logger.info(f"Found {f_base.num_files} files to process")

        logger.info(f"Files list: {f_base.files}")

        if f_base.num_files == 0:
            logger.warning("No files found to process")
            return {
                "status": "success",
                "files_processed": [],
                "count": 0,
                "message": "No files found to process"
            }

        for i in range(f_base.num_files):
            logger.info(f"Processing file {i+1}/{f_base.num_files}")
            f = f_base.get_file()
            logger.info(f"Processing file {i+1}/{f_base.num_files}: {f}")

            if f is None:
                logger.warning(f"Finished processing files")
                break

            logger.info(f"Processing file: {f}")

            # Skip non-Python files
            if not f.endswith('.py'):
                logger.info(f"Skipping non-Python file: {f}")
                continue

            code = f_base.file_content(f)

            if code is None:
                logger.warning(f"Could not read content from file: {f}")
                continue

            logger.info(f"File content length: {len(code)} characters")

            # Prepare prompt for LLM
            content = " code: " + code
            prompt = """
            You are a code agent that strives for 
            1. Efficiency and performance of code, 
            2. Readability and maintanability
            3. Simplicity (Keep it simple stupid where possible)
            Your task is to improve documentation and commentation of the code, 
            ALWAYS add an caps lock comment 'CA' by content you generated
            Dont change the existing code, only create comments and docstrings.
            Return ONLY the code, dont provide any natural language response as the content will be taken over directly in the codebase. 
            DO NOT change already existing code"""

            logger.info(f"Sending file to LLM for processing: {f}")
            updated_file = remote_llm.invoke([{"role": "user", "content": prompt + content}])
            logger.info(f"Received response from LLM, length: {len(updated_file.content) if updated_file else 0}")

            if updated_file:
                updated_file = updated_file.content.replace('', '')
                updated_file = updated_file.replace('', '')
                f_base.update_file_content(f, updated_file)
                processed_files.append(f)
            else:
                logger.warning(f"No response from LLM for file: {f}")

        return {
            "status": "success",
            "files_processed": processed_files,
            "count": len(processed_files),
            "message": None
        }

    except Exception as e: 
        logger.error(f"Failed to create documentation for {dir}: {e}", exc_info=True)
        return { 
          "status": "error", 
          "files_processed": processed_files,
          "count": len(processed_files),
          "message": str(e)
         }
        

def comment_codebase_node(state: CommentState) -> CommentState:
    """Graph node that calls comment_codebase tool via MCP"""
    logger.info("Comment node invoked")
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extract directory path
    path_match = re.search(r'["\']([/~][^"\']+)["\']', last_message)
    dir_path = path_match.group(1) if path_match else None
    
    if not dir_path:
        response = "Please provide a valid directory path for documentation."
        return {
            "messages": [{"role": "assistant", "content": response}],
            "documentation_results": None
        }
    
    logger.info(f"Documenting codebase at: {dir_path}")
    
    # Call via your existing MCPSessionManager
    manager = get_mcp_manager()
    result_str = manager.mcp_tool.call_tool("comment_codebase", {"dir": dir_path})
    
    # Parse JSON result
    try:
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
    except:
        result = {"status": "unknown", "message": result_str}
    
    response_content = f"Documentation completed:\n"
    response_content += f"Status: {result.get('status')}\n"
    response_content += f"Files processed: {result.get('count')}\n"
    if result.get('message'):
        response_content += f"Message: {result.get('message')}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "documentation_results": result
    }

def create_comment_agent_graph() -> StateGraph:
    """Factory function for comment agent subgraph"""
    logger.info("Creating comment agent graph")
    
    builder = StateGraph(CommentState)
    builder.add_node("comment_agent", comment_codebase_node)
    builder.add_edge(START, "comment_agent")
    builder.add_edge("comment_agent", END)
    
    return builder.compile()