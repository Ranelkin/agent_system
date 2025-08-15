from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import os, json, asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict
from src.model.agent import llm

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class MCPTool:
    def __init__(self, server_command):
        self.server_command = server_command
        self.module_path = "src.mcp.mcp_server"

    def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        return asyncio.run(self._async_call_mcp_tool(tool_name, args))

    async def _async_call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        server_params = StdioServerParameters(
            command="python",
            args=["-m", self.module_path],
            env={}
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=args)
                    return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"

    def search(self, query: str) -> str:
        return self._call_mcp_tool("search", {"search_term": query})

    def document_codebase(self, directory: str) -> str:
        return self._call_mcp_tool("document_codebase", {"dir": directory})

# Create tools
mcp_tool = MCPTool(["python3", "-m", "src.mcp.mcp_server"])
tools = [
    Tool(
        name="web_search",
        func=mcp_tool.search,
        description="Search the web for information"
    ),
    Tool(
        name="document_codebase",
        func=mcp_tool.document_codebase,
        description="Generate documentation for a codebase in the specified directory"
    )
]

# Create ReAct agent with LangGraph
system_prompt = """You are an AI assistant that can search the web or document codebases. 
For queries requesting to document a codebase, use the Document Codebase tool with the provided directory. 
For other queries, use the Web Search tool if applicable. Respond with the tool's output."""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{{input}}")
])
agent = create_react_agent(llm, tools, prompt=prompt)

# Chat loop
def chat_with_agent():
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = agent.invoke({"input": query})
        print(response["output"])

if __name__ == "__main__":
    chat_with_agent()