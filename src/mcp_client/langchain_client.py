import os, json, logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict, Optional
import nest_asyncio

# Allow nested event loops (useful for Jupyter/interactive environments)
nest_asyncio.apply()
logger = logging.getLogger("session_manager")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class MCPTool:
    def __init__(self, server_command):
        self.server_command = server_command
        # Use the mcp_server module path correctly
        self.mcp_server = StdioServerParameters(
            command="python3",
            args=["-m", "src.mcp_server.mcp_server"],
            env=None  # Use None instead of empty dict
        )
        self.session: Optional[ClientSession] = None
        self.read = None
        self.write = None
        self.client_context = None
        
    async def __aenter__(self):
        """Async context manager entry - establishes connection"""
        logger.info("Starting MCP client connection...")
        self.client_context = stdio_client(self.mcp_server)
        self.read, self.write = await self.client_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        logger.info("MCP client connection established")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleans up connection"""
        logger.info("Closing MCP client connection...")
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self.client_context:
            await self.client_context.__aexit__(exc_type, exc_val, exc_tb)
        logger.info("MCP client connection closed")
            
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call MCP tool with existing session"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with MCPTool(...) as mcp:'")
        try:
            logger.debug(f"Calling tool {tool_name} with args: {args}")
            result = await self.session.call_tool(tool_name, arguments=args)
            logger.debug(f"Tool {tool_name} returned: {result}")
            return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {str(e)}")
            raise

