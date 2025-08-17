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
        self.module_path = "src.mcp_server.mcp_server"
        self.mcp_server = StdioServerParameters(
            command="python3",
            args=["-m", self.module_path],
            env={}
        )
        
    def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        return asyncio.run(self._async_call_mcp_tool(tool_name, args))

    async def _async_call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            async with stdio_client(self.mcp_server) as (read, write):
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
mcp_tool = MCPTool(["python3", "-m", "src.mcp_server.mcp_server"])
tools = [
    Tool(
        name="web_search",
        func=mcp_tool.search,
        description="Search the web for information"
    ),
    Tool(
        name="document_codebase",
        func=mcp_tool.document_codebase,
        description="Generate documentation for a codebase. Pass the full directory path as the argument."
    )
]

# Create ReAct agent with LangGraph
system_prompt = """You are an AI assistant that can search the web or document codebases.

When asked to document a codebase:
1. Extract the exact directory path from the user's message
2. Call the document_codebase tool with that exact path as the argument
3. Return the tool's output

For other queries, use the Web Search tool if applicable."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{{input}}")  
])

agent = create_react_agent(
    llm, 
    tools, 
    prompt=prompt
)

# Chat loop
def chat_with_agent():
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        try:
            # Add recursion_limit config to prevent infinite loops
            response = agent.invoke(
                {"messages": [("human", query)]},
                config={"recursion_limit": 10}
            )
            # Extract the final message from the response
            if "messages" in response:
                print(response["messages"][-1].content)
            else:
                print(response)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_agent()