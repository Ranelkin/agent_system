from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import asyncio, logging
from typing import  Optional
from .langchain_client import MCPTool
from model.agent import llm

logger = logging.getLogger("session_manager")

class MCPSessionManager:
    def __init__(self):
        self.mcp_tool: Optional[MCPTool] = None
        
    async def start(self):
        """Start persistent connection"""
        logger.info("Starting MCPSessionManager...")
        self.mcp_tool = MCPTool(["python3", "-m", "src.mcp_server.mcp_server"])
        await self.mcp_tool.__aenter__()
        return self.mcp_tool
    
    async def stop(self):
        """Stop persistent connection"""
        logger.info("Stopping MCPSessionManager...")
        if self.mcp_tool:
            await self.mcp_tool.__aexit__(None, None, None)
            
    def get_tools(self):
        """Get LangChain tools with persistent connection"""
        if not self.mcp_tool:
            raise RuntimeError("MCP session not started. Call start() first.")
        
        # Create wrapper functions that use the persistent connection
        def search_sync(query: str) -> str:
            """Synchronous wrapper for search using persistent connection"""
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.mcp_tool.call_tool("search", {"search_term": query}))
                    return future.result()
            else:
                return asyncio.run(self.mcp_tool.call_tool("search", {"search_term": query}))
            
        def document_sync(directory: str) -> str:
            """Synchronous wrapper for document_codebase using persistent connection"""
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.mcp_tool.call_tool("document_codebase", {"dir": directory}))
                    return future.result()
            else:
                return asyncio.run(self.mcp_tool.call_tool("document_codebase", {"dir": directory}))
        
        return [
            Tool(
                name="web_search",
                func=search_sync,
                description="Search the web for information"
            ),
            Tool(
                name="document_codebase", 
                func=document_sync,
                description="Generate documentation for a codebase. Pass the full directory path as the argument."
            )
        ]

async def chat_session():
    """Main chat session with persistent MCP connection"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    manager = MCPSessionManager()
    await manager.start()
    
    try:
        tools = manager.get_tools()
        
        system_prompt = """You are an AI assistant that can search the web or document codebases.

        When asked to document a codebase:
        1. Extract the exact directory path from the user's message
        2. Call the document_codebase tool with that exact path as the argument
        3. Return the tool's output

        For other queries, use the Web Search tool if applicable."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{{input}}")  # Use single braces for LangChain
        ])

        agent = create_react_agent(
            llm, 
            tools, 
            prompt=prompt
        )
        
        # Chat loop
        logger.info("\n🤖 AI Assistant ready! (type 'exit' to quit)\n")
        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            logger.info("\nAssistant: ")
            try:
                response = await agent.ainvoke(
                    {"messages": [("human", query)]},
                    config={"recursion_limit": 10}
                )
                
                # Extract the final message
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1]
                    if hasattr(final_message, 'content'):
                        logger.info(final_message.content)
                    else:
                        logger.info(final_message)
                else:
                    logger.info(response)
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                logger.info(f"Error: {e}")
            
            logger.info()  # Add spacing between conversations
                
    finally:
        await manager.stop()
        logger.info("\n👋 Goodbye!")

# Run the chat session
if __name__ == "__main__":
    try:
        asyncio.run(chat_session())
    except KeyboardInterrupt:
        logger.info("\n\n👋 Chat interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info(f"\n❌ Fatal error: {e}")