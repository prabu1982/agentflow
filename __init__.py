"""Mainframe Agent - Automation agent with LangGraph, MCP Server, and OLLAMA error recovery."""

from .mainframe_agent import MainframeAgent
from .config import AgentConfig
from .mcp_client import MCPClient
from .ollama_client import OllamaClient
from .step_reader import StepReader

__version__ = "1.0.0"
__all__ = [
    "MainframeAgent",
    "AgentConfig",
    "MCPClient",
    "OllamaClient",
    "StepReader"
]
