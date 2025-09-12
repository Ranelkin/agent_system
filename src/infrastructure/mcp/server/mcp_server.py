from fastmcp import FastMCP
from ....shared.log_config import setup_logging
from ....core.services.grouped_tools import get_available_tools
logger = setup_logging("mcp_server")

mcp = FastMCP(
    name="AAS",
    instructions="""
        This server provides different mcp tools.
    """,
)

try: 
    tools = get_available_tools()

    for func in tools: 
        mcp.tool(
            func, 
            name=None, 
            description=None
        )
except Exception as e: 
    logger.error(f"Failed to register tools: {e}")

if __name__ == '__main__': 
    mcp.run()