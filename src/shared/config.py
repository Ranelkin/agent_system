from pydantic import BaseSettings

class AppConfig(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    log_level: str = "INFO"
    mcp_server_timeout: int = 30
    
    class Config:
        env_file = ".env"

config = AppConfig()