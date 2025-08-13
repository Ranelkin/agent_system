from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os, json 
from ..model.agent import llm 
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class MCPTool:
    def __init__(self, server_command):
        self.server_command = server_command
    
    def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Synchronous wrapper for async MCP tool calls"""
        return asyncio.run(self._async_call_mcp_tool(tool_name, args))
    
    async def _async_call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Async method to call MCP tools"""
        server_params = StdioServerParameters(
            command="python",
            args=["-m", self.module_path],
            env={}
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments=args)
                    
                    # Return the result as a string
                    if isinstance(result, dict):
                        return json.dumps(result, indent=2)
                    return str(result)
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"
    
    
    def search(self, query: str) -> str:
        # Call your MCP server's search function
        result = self._call_mcp_tool("search", {"search_term": query})
        return result
    
    def document_codebase(self, directory: str) -> str:
        # Call your MCP server's document_codebase function
        result = self._call_mcp_tool("document_codebase", {"dir": directory})
        return result
    
    
    

# Create LangChain tools
mcp_tool = MCPTool(["python3", "-m", "..mcp.mcp_server"])

tools = [
    Tool(
        name="Web Search",
        func=mcp_tool.search,
        description="Search the web for information"
    ),
    Tool(
        name="Document Codebase",
        func=mcp_tool.document_codebase,
        description="Generate documentation for a codebase"
    )
]

# Initialize an agent with these tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

