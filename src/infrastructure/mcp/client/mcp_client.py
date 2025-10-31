import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict, Optional
from ....shared.log_config import setup_logging
logger = setup_logging("session_manager")

class MCPTool:
    def __init__(self, server_command):
        """Initialize MCPTool with server command details and setup for connection."""
        self.server_command = server_command
        self.mcp_server = StdioServerParameters(
            command="python3",
            args=["-m", "src.infrastructure.mcp.server.mcp_server"],
            env=None  
        )
        self.session: Optional[ClientSession] = None
        self.read = None
        self.write = None
        self.client_context = None
        self.loop = None
        
    async def _async_connect(self): 
        """Async connection to MCP server"""
        self.client_context = stdio_client(self.mcp_server)
        self.read, self.write = await self.client_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        logger.info("MCP client connected")
        
    def connect(self):
        """Synchronous connection wrapper"""
        try: 
            self.loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError: 
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        self.loop.run_until_complete(self._async_connect())
        logger.info("MCP client connection established")
        
    async def _async_disconnect(self): 
        """Async disconnection"""
        if self.session: 
            await self.session.__aexit__(None, None, None)
        if self.client_context: 
            await self.client_context.__aexit__(None, None, None)
        logger.info("MCP client disconnected")
        
    def disconnect(self):
        """Synchronous disconnection wrapper"""
        try: 
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError: 
            loop = asyncio.get_event_loop()
            
        loop.run_until_complete(self._async_disconnect())
    
    async def _async_call_tool(self, tool_name: str, args: Dict[str, Any]) -> str: 
        """Asynchronous tool call wrapper"""
        result = await self.session.call_tool(tool_name, arguments=args)
        import json 
        json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Synchronous tool call wrapper"""
        try: 
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self._async_call_tool(tool_name, args))
        except RuntimeError: 
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._async_call_tool(tool_name, args=args))
    
    


 