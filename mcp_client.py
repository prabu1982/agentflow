"""MCP Server client for executing automation steps.
Supports both HTTP/SSE and stdio transport modes with comprehensive exception handling."""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from config import AgentConfig

logger = logging.getLogger(__name__)

# FastMCP client for HTTP transport
try:
    from fastmcp import Client as FastMCPClient
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logger.warning("fastmcp not available. Install with: pip install fastmcp")

# Optional MCP SDK support (not required for HTTP transport)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # This is fine - we use fastmcp for HTTP transport


# Custom Exception Classes
class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPClientError):
    """Raised when connection to MCP server fails."""
    pass


class MCPTimeoutError(MCPClientError):
    """Raised when MCP request times out."""
    pass


class MCPToolError(MCPClientError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, message: str, error_details: Optional[Dict[str, Any]] = None):
        self.tool_name = tool_name
        self.error_details = error_details or {}
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class MCPServerError(MCPClientError):
    """Raised when MCP server returns an error."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.response = response or {}
        super().__init__(message)


class MCPHttpClient:
    """MCP Client using HTTP/SSE transport via FastMCP."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize HTTP MCP client using FastMCP.
        
        Args:
            config: Agent configuration with mcp_server_url
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("fastmcp is required for HTTP transport. Install with: pip install fastmcp")
        
        self.config = config
        self.base_url = config.mcp_server_url.rstrip('/') if config.mcp_server_url else ""
        self.timeout = config.mcp_timeout
        self.client: Optional[FastMCPClient] = None
        self._initialized = False
        self._connection_healthy = False
    
    async def _initialize(self):
        """Initialize the FastMCP client session."""
        if self._initialized:
            return
        
        try:
            if not self.base_url:
                raise MCPConnectionError("MCP server URL is not configured")
            
            logger.info(f"Initializing FastMCP client for URL: {self.base_url}")
            
            # FastMCP Client expects the full URL including /mcp endpoint if needed
            # Check if /mcp is already in the URL, if not add it
            client_url = self.base_url
            if not client_url.endswith('/mcp') and '/mcp' not in client_url:
                # FastMCP typically expects /mcp endpoint
                client_url = f"{self.base_url}" if not self.base_url.endswith('/') else f"{self.base_url}mcp"
            
            self.client = FastMCPClient(client_url)
            
            # Initialize the client connection (enter context manager)
            await self.client.__aenter__()
            
            self._initialized = True
            self._connection_healthy = True
            logger.info("FastMCP HTTP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FastMCP client: {str(e)}", exc_info=True)
            raise MCPConnectionError(f"Failed to initialize FastMCP client: {str(e)}") from e
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool using HTTP transport via FastMCP.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result with success status
            
        Raises:
            MCPToolError: If tool execution fails
            MCPConnectionError: If connection issues occur
            MCPTimeoutError: If request times out
        """
        if not self._initialized:
            await self._initialize()
        
        if not self._connection_healthy:
            raise MCPConnectionError("MCP server connection is not healthy")
        
        try:
            logger.info(f"Calling MCP tool via FastMCP HTTP: {tool_name} with arguments: {arguments}")
            
            # Use FastMCP client's call_tool method
            result = await self.client.call_tool(tool_name, arguments)
            
            # FastMCP returns result objects, convert to dict if needed
            if hasattr(result, 'content'):
                # Extract content from FastMCP result
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    # Get the first content item
                    first_content = content[0]
                    if hasattr(first_content, 'text'):
                        result_dict = {"text": first_content.text}
                    elif isinstance(first_content, dict):
                        result_dict = first_content
                    else:
                        result_dict = {"content": str(first_content)}
                else:
                    result_dict = {"content": str(content)} if content else {}
            elif hasattr(result, 'text'):
                result_dict = {"text": result.text}
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {"result": str(result)}
            
            logger.info(f"MCP tool {tool_name} executed successfully via FastMCP HTTP")
            return {
                "success": True,
                "tool": tool_name,
                "result": result_dict,
                "message": "Tool executed successfully"
            }
            
        except Exception as e:
            # Check error type
            error_type = type(e).__name__
            error_msg = str(e)
            
            if "Timeout" in error_type or "timeout" in error_msg.lower():
                error_msg = f"Request timeout calling MCP tool {tool_name}: {error_msg}"
                logger.error(error_msg)
                self._connection_healthy = False
                raise MCPTimeoutError(error_msg) from e
            elif "Connection" in error_type or "connection" in error_msg.lower() or "Connect" in error_type:
                error_msg = f"Connection error calling MCP tool {tool_name}: {error_msg}"
                logger.error(error_msg)
                self._connection_healthy = False
                raise MCPConnectionError(error_msg) from e
            elif "HTTP" in error_type or "status" in error_msg.lower():
                # HTTP error from server
                status_code = None
                if "status" in error_msg.lower():
                    # Try to extract status code
                    import re
                    match = re.search(r'(\d{3})', error_msg)
                    if match:
                        status_code = int(match.group(1))
                raise MCPServerError(
                    f"HTTP error calling MCP tool {tool_name}: {error_msg}",
                    status_code=status_code,
                    response={"error": error_msg}
                )
            else:
                # Other errors
                error_msg = f"Error calling MCP tool {tool_name} via FastMCP: {error_msg}"
                logger.error(error_msg, exc_info=True)
                raise MCPToolError(tool_name, error_msg, {"exception": str(e)}) from e
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            if not self._initialized:
                await self._initialize()
            
            # Use FastMCP client's list_tools method
            tools = await self.client.list_tools()
            
            # Convert FastMCP tool objects to dictionaries
            tools_list = []
            for tool in tools:
                tool_dict = {
                    "name": tool.name if hasattr(tool, 'name') else str(tool),
                    "description": tool.description if hasattr(tool, 'description') else "",
                }
                if hasattr(tool, 'inputSchema'):
                    tool_dict["inputSchema"] = tool.inputSchema
                tools_list.append(tool_dict)
            
            return tools_list
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}", exc_info=True)
            raise MCPConnectionError(f"Failed to list tools: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the MCP server connection is healthy."""
        try:
            if not self._initialized:
                return False
            
            # Try to list tools as a health check
            await self.list_tools()
            self._connection_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            self._connection_healthy = False
            return False
    
    async def close(self):
        """Close the FastMCP client."""
        self._connection_healthy = False
        if self.client and self._initialized:
            try:
                # Exit the context manager
                await self.client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing FastMCP client: {e}")
        self._initialized = False
        self.client = None


class MCPClient:
    """Unified MCP Client supporting both HTTP/SSE and stdio transports with comprehensive exception handling."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize MCP client with appropriate transport.
        
        Args:
            config: Agent configuration
            
        Raises:
            ValueError: If required configuration is missing
            MCPConnectionError: If initialization fails
        """
        self.config = config
        self.transport = config.mcp_transport.lower()
        
        # Initialize appropriate client based on transport
        try:
            if self.transport == "stdio":
                if not config.mcp_server_command:
                    raise ValueError("mcp_server_command is required for stdio transport")
                self.client = MCPStdioClient(config)
                logger.info("Initialized MCP client with stdio transport")
            else:
                if not config.mcp_server_url:
                    raise ValueError("mcp_server_url is required for HTTP transport")
                self.client = MCPHttpClient(config)
                logger.info(f"Initialized MCP client with HTTP transport (URL: {config.mcp_server_url})")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {str(e)}", exc_info=True)
            raise MCPConnectionError(f"Failed to initialize MCP client: {str(e)}") from e
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with retry logic.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments for the tool
            retries: Number of retries on failure (default: 0)
            
        Returns:
            Tool execution result with success status
            
        Raises:
            MCPToolError: If tool execution fails after retries
            MCPConnectionError: If connection issues occur
            MCPTimeoutError: If request times out
        """
        if not tool_name:
            raise ValueError("tool_name is required")
        
        last_error = None
        for attempt in range(retries + 1):
            try:
                result = await self.client.call_tool(tool_name, arguments)
                
                # If successful, return immediately
                if result.get("success"):
                    return result
                
                # If not successful but no exception, log and retry
                if attempt < retries:
                    logger.warning(
                        f"Tool {tool_name} failed (attempt {attempt + 1}/{retries + 1}): "
                        f"{result.get('error', 'Unknown error')}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                
                # Last attempt failed
                error_msg = result.get("error", "Unknown error")
                raise MCPToolError(tool_name, error_msg, result)
                
            except (MCPConnectionError, MCPTimeoutError) as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        f"Connection error calling {tool_name} (attempt {attempt + 1}/{retries + 1}): {str(e)}"
                    )
                    # Try to reconnect
                    try:
                        await self.client.health_check()
                    except:
                        pass
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except MCPToolError:
                raise
            except Exception as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        f"Unexpected error calling {tool_name} (attempt {attempt + 1}/{retries + 1}): {str(e)}"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                raise MCPToolError(tool_name, f"Unexpected error: {str(e)}", {"exception": str(e)}) from e
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise MCPToolError(tool_name, "Tool execution failed after retries")
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a step by calling the appropriate MCP tool.
        
        Args:
            step: Step definition with tool name and parameters
            
        Returns:
            Step execution result with success status
            
        Raises:
            ValueError: If step is missing required fields
            MCPToolError: If tool execution fails
        """
        tool_name = step.get("tool")
        if not tool_name:
            return {
                "success": False,
                "error": "Step missing 'tool' field",
                "message": "Step definition must include a 'tool' field"
            }
        
        arguments = step.get("arguments", step.get("params", {}))
        
        # Use retry logic from config
        retries = getattr(self.config, 'max_retries', 0)
        
        try:
            return await self.call_tool(tool_name, arguments, retries=retries)
        except (MCPToolError, MCPConnectionError, MCPTimeoutError) as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"Tool execution failed: {str(e)}"
            }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of available tools
            
        Raises:
            MCPConnectionError: If connection fails
        """
        try:
            return await self.client.list_tools()
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}", exc_info=True)
            raise MCPConnectionError(f"Failed to list tools: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            return await self.client.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the MCP client."""
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing MCP client: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
