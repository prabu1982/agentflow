"""MCP Server client for executing automation steps."""

import logging
import httpx
from typing import Dict, Any, Optional
from config import AgentConfig

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with MCP Server."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize MCP client.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.base_url = config.mcp_server_url.rstrip('/')
        self.timeout = config.mcp_timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            Exception: If tool execution fails
        """
        try:
            logger.info(f"Calling MCP tool: {tool_name} with arguments: {arguments}")
            
            # MCP Server typically uses POST /tools/{tool_name} or similar
            # Adjust endpoint based on your MCP server implementation
            endpoint = f"/tools/{tool_name}"
            
            response = await self.client.post(
                endpoint,
                json={
                    "tool": tool_name,
                    "arguments": arguments
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"MCP tool {tool_name} executed successfully")
            return {
                "success": True,
                "tool": tool_name,
                "result": result,
                "message": "Tool executed successfully"
            }
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error calling MCP tool {tool_name}: {e.response.text}"
            logger.error(error_msg)
            return {
                "success": False,
                "tool": tool_name,
                "error": error_msg,
                "status_code": e.response.status_code,
                "message": f"Tool execution failed: {error_msg}"
            }
            
        except httpx.RequestError as e:
            error_msg = f"Request error calling MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "tool": tool_name,
                "error": error_msg,
                "message": f"Failed to connect to MCP server: {error_msg}"
            }
            
        except Exception as e:
            error_msg = f"Unexpected error calling MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "tool": tool_name,
                "error": error_msg,
                "message": f"Unexpected error: {error_msg}"
            }
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a step by calling the appropriate MCP tool.
        
        Args:
            step: Step definition with tool name and parameters
            
        Returns:
            Step execution result
        """
        tool_name = step.get("tool")
        if not tool_name:
            return {
                "success": False,
                "error": "Step missing 'tool' field",
                "message": "Step definition must include a 'tool' field"
            }
        
        arguments = step.get("arguments", step.get("params", {}))
        
        return await self.call_tool(tool_name, arguments)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
