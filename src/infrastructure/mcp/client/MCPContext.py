from .session_manager import MCPSessionManager
from typing import Optional

class MCPContext:
    """Global singleton wrapper for MCPSessionManager"""
    _instance: Optional[MCPSessionManager] = None
    
    @classmethod
    def get_instance(cls) -> MCPSessionManager:
        """Get or create the global MCP manager instance"""
        if cls._instance is None:
            cls._instance = MCPSessionManager()
            cls._instance.start()
        return cls._instance
    
    @classmethod
    def shutdown(cls):
        """Shutdown the global MCP manager"""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None


def get_mcp_manager() -> MCPSessionManager:
    """Get the global MCP manager"""
    return MCPContext.get_instance()

def shutdown_mcp():
    """Shutdown MCP connection"""
    MCPContext.shutdown()