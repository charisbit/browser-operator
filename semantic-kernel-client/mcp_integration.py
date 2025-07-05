"""
MCP Integration Module for Semantic Kernel
Integration with Microsoft's official playwright-mcp
"""
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_plugin import KernelPlugin

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPTransport:
    """MCP transport base class"""
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
    async def close(self):
        pass

class StdioTransport(MCPTransport):
    """Standard I/O transport"""
    def __init__(self, command: List[str]):
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    async def connect(self) -> bool:
        """Connects to the MCP server"""
        try:
            logger.info(f"Starting MCP server: {' '.join(self.command)}")
            
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "semantic-kernel-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self.send_request(init_request)
            if response and not response.get("error"):
                logger.info("✅ MCP connection established successfully")
                
                # Send initialized notification
                await self._send_notification({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                })
                
                return True
            else:
                logger.error(f"❌ MCP initialization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MCP connection error: {e}")
            return False
    
    async def send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends a request and waits for a response"""
        if not self.process:
            return None
            
        try:
            # Send the request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            # Read the response
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Request/response error: {e}")
            return None
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Sends a notification (no response needed)"""
        if not self.process:
            return
            
        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"❌ Notification error: {e}")
    
    def _get_request_id(self) -> int:
        """Generates a request ID"""
        self.request_id += 1
        return self.request_id
    
    async def close(self):
        """Closes the connection"""
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                logger.info("🔒 MCP connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing connection: {e}")
            finally:
                self.process = None

class SSETransport(MCPTransport):
    """SSE (Server-Sent Events) transport"""
    def __init__(self, url: str):
        self.url = url
        self.session = None
        
    async def connect(self) -> bool:
        """Connects to the SSE endpoint"""
        import aiohttp
        
        try:
            self.session = aiohttp.ClientSession()
            # SSE connection logic
            logger.info(f"Connecting to SSE: {self.url}")
            return True
        except Exception as e:
            logger.error(f"SSE connection failed: {e}")
            return False
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a request via SSE"""
        # SSE request implementation
        pass
    
    async def close(self):
        """Closes the SSE connection"""
        if self.session:
            await self.session.close()

class PlaywrightMCPClient:
    """Playwright MCP client"""
    
    def __init__(self, transport: Union[str, List[str]] = None):
        """
        Initializes the MCP client
        
        Args:
            transport: Transport configuration
                - List[str]: Command-line arguments, uses stdio transport
                - str: URL, uses SSE transport
                - None: Defaults to npx @playwright/mcp@latest
        """
        if transport is None:
            transport = ["npx", "@playwright/mcp@latest"]
        
        if isinstance(transport, list):
            self.transport = StdioTransport(transport)
        elif isinstance(transport, str):
            self.transport = SSETransport(transport)
        else:
            raise ValueError("transport must be a list of commands or a URL string")
        
        self.tools: List[MCPTool] = []
        self.connected = False
    
    async def connect(self) -> bool:
        """Connects to the MCP server"""
        self.connected = await self.transport.connect()
        if self.connected:
            await self._load_tools()
        return self.connected
    
    async def _load_tools(self):
        """Loads the available tools"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = await self.transport.send_request(request)
            if response and not response.get("error"):
                tools_data = response.get("result", {}).get("tools", [])
                self.tools = [
                    MCPTool(
                        name=tool["name"],
                        description=tool["description"],
                        inputSchema=tool["inputSchema"]
                    )
                    for tool in tools_data
                ]
                
                logger.info(f"📋 Loaded {len(self.tools)} tools")
                for tool in self.tools:
                    logger.debug(f"  🔧 {tool.name}: {tool.description}")
            else:
                logger.error(f"❌ Failed to load tools: {response}")
                
        except Exception as e:
            logger.error(f"❌ Exception while loading tools: {e}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Calls a tool"""
        try:
            logger.info(f"🔧 Calling tool: {name} with arguments: {arguments}")
            
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
            
            response = await self.transport.send_request(request)
            if response and not response.get("error"):
                result = response.get("result", {})
                logger.info(f"✅ Tool {name} executed successfully")
                return {
                    "success": True,
                    "content": result.get("content", []),
                    "isError": result.get("isError", False)
                }
            else:
                error = response.get("error", {}) if response else {"message": "No response"}
                logger.error(f"❌ Tool {name} execution failed: {error}")
                return {
                    "success": False,
                    "error": error.get("message", "Unknown error"),
                    "code": error.get("code", -1)
                }
                
        except Exception as e:
            logger.error(f"❌ Exception while calling tool {name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Closes the client"""
        await self.transport.close()
        self.connected = False

class PlaywrightMCPPlugin(KernelPlugin):
    """Playwright MCP plugin for Semantic Kernel"""
    
    client: Optional[PlaywrightMCPClient] = None
    
    def __init__(self, client: PlaywrightMCPClient, name: str = "PlaywrightBrowser"):
        super().__init__(name=name, description="Playwright browser automation plugin")
        self.client = client
        self._register_functions()
    
    @classmethod
    async def create(cls, transport: Union[str, List[str]] = None, name: str = "PlaywrightBrowser"):
        """Creates and initializes the plugin"""
        client = PlaywrightMCPClient(transport)
        if not await client.connect():
            raise RuntimeError("Could not connect to the playwright-mcp server")
        return cls(client, name)
    
    def _register_functions(self):
        """Registers all MCP tools as Kernel functions"""
        for tool in self.client.tools:
            # Dynamically create a function
            func = self._create_tool_function(tool)
            # Add the function to the plugin
            self[tool.name] = func
    
    def _create_tool_function(self, tool: MCPTool):
        """Creates a Kernel function for an MCP tool"""
        async def tool_function(arguments) -> str:
            result = await self.client.call_tool(tool.name, arguments)
            
            if result.get("success"):
                # Format the content
                content = result.get("content", [])
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return "\n".join(text_parts)
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        
        # Set function metadata
        tool_function.__name__ = tool.name
        tool_function.__doc__ = tool.description
        
        # Use the kernel_function decorator
        return kernel_function(
            name=tool.name,
            description=tool.description
        )(tool_function)
    
    async def close(self):
        """Closes the plugin"""
        await self.client.close()

# Convenience function
async def create_playwright_plugin(transport: Union[str, List[str]] = None) -> PlaywrightMCPPlugin:
    """
    Creates a Playwright MCP plugin
    
    Args:
        transport: Transport configuration
            - None: Uses the default npx @playwright/mcp@latest
            - List[str]: Custom command
            - str: SSE URL
    
    Returns:
        PlaywrightMCPPlugin: An initialized plugin
    
    Example:
        ```python
        # Use default configuration
        plugin = await create_playwright_plugin()
        
        # Use a custom command
        plugin = await create_playwright_plugin(["node", "my-mcp-server.js"])
        
        # Use SSE
        plugin = await create_playwright_plugin("http://localhost:8931/sse")
        ```
    """
    return await PlaywrightMCPPlugin.create(transport)
