"""Configuration management for Mainframe Agent."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class AgentConfig(BaseSettings):
    """Configuration for Mainframe Agent."""
    
    # LLM Configuration
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="OLLAMA LLM server URL for error fixing"
    )
    ollama_model: str = Field(
        default="llama3",
        description="OLLAMA model name to use"
    )
    
    # Google AI SDK Configuration (Optional - not currently used)
    # Reserved for future use if needed for advanced LLM features
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google AI API key for Gemini (optional, not currently used)"
    )
    google_model: str = Field(
        default="gemini-pro",
        description="Google Gemini model name (optional, not currently used)"
    )
    
    # MCP Server Configuration
    mcp_server_url: str = Field(
        default="http://localhost:8000",
        description="MCP Server URL for step execution"
    )
    mcp_timeout: int = Field(
        default=30,
        description="MCP Server request timeout in seconds"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed steps"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            if field_name in ["max_retries", "mcp_timeout"]:
                return int(raw_val)
            if field_name == "retry_delay":
                return float(raw_val)
            return cls.json_loads(raw_val)
    
    def __init__(self, **kwargs):
        # Load from environment variables if not provided
        super().__init__(**kwargs)
        
        # Override with environment variables if set
        self.ollama_url = os.getenv("OLLAMA_URL", self.ollama_url)
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.google_model = os.getenv("GOOGLE_MODEL", self.google_model)
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", self.mcp_server_url)
        self.mcp_timeout = int(os.getenv("MCP_TIMEOUT", str(self.mcp_timeout)))
        self.max_retries = int(os.getenv("MAX_RETRIES", str(self.max_retries)))
        self.retry_delay = float(os.getenv("RETRY_DELAY", str(self.retry_delay)))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
