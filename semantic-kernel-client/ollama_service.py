"""
Ollama Service for Semantic Kernel - Official API Compatible Version
Implementation that fully complies with the official Ollama API documentation
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
from dataclasses import dataclass
import yaml
import os

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Ollama configuration"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:function-calling"  # Default to a model that supports function calling
    temperature: float = 0.0
    timeout: int = 120
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

class OllamaPromptExecutionSettings(PromptExecutionSettings):
    """Ollama-specific execution settings"""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    tfs_z: Optional[float] = None
    format: Optional[str] = None  # Supports 'json' or a JSON schema
    
class OllamaChatCompletion(ChatCompletionClientBase):
    """Ollama chat completion service - Official API compatible version"""
    
    model_id: str = "llama3.2:function-calling"
    base_url: str = "http://localhost:11434"
    service_id: str = "ollama_chat"
    timeout: int = 120
    
    def __init__(
        self,
        model_id: str = "llama3.2:function-calling",
        base_url: str = "http://localhost:11434",
        service_id: str = "ollama_chat",
        timeout: int = 120,  # Add timeout parameter
    ):
        """
        Initializes the Ollama chat service
        
        Args:
            model_id: Ollama model ID
            base_url: Ollama API base URL
            service_id: Service ID
        """
        super().__init__(service_id=service_id, ai_model_id=model_id)
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # Save timeout setting
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensures that an HTTP session exists"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs
    ) -> List[ChatMessageContent]:
        """
        Completes a chat request
        
        Args:
            chat_history: The chat history
            settings: The execution settings
            
        Returns:
            A list of chat message content
        """
        await self._ensure_session()
        
        # Convert settings
        options = self._convert_settings(settings)
        
        # Build the request
        messages = self._format_chat_history(chat_history)
        
        # Build the request data
        request_data = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        # Add function calling support (if tools are provided)
        tools = kwargs.get("tools", [])
        if tools:
            # Convert tool format to the official API format
            request_data["tools"] = self._format_tools_for_api(tools)
        
        # Add format parameter (if in settings)
        if isinstance(settings, OllamaPromptExecutionSettings) and settings.format:
            request_data["format"] = settings.format
        
        # Add keep_alive parameter
        if "keep_alive" in kwargs:
            request_data["keep_alive"] = kwargs["keep_alive"]
        
        try:
            logger.info(f"🤖 Sending request to Ollama, model: {self.model_id}")
            logger.info(f"   🕑 Timeout setting: {self.timeout}s")
            logger.debug(f"Request data: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
            
            async with self._session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Response data: {result}")
                    message = result.get("message", {})
                    
                    logger.info("✅ Received response from Ollama")
                    
                    # Extract content and tool calls
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    
                    # If there are no standard tool_calls, try to parse the functools format from the content
                    if not tool_calls and content:
                        parsed_content, parsed_calls = self._parse_functools_format(content)
                        if parsed_calls:
                            tool_calls = parsed_calls
                            content = parsed_content  # Use the cleaned content
                    
                    # If there are tool calls, process them
                    if tool_calls and "kernel" in kwargs:
                        logger.info(f"🔧 Detected {len(tool_calls)} tool calls")
                        for tc in tool_calls:
                            if "function" in tc:
                                func = tc["function"]
                                logger.info(f"  📦 Function: {func.get('name')} with arguments: {func.get('arguments')}")
                        
                        # Execute the tool calls
                        tool_results = await self._execute_tool_calls(
                            tool_calls, 
                            kwargs.get("kernel")
                        )
                        
                        # Build a response containing the tool execution results
                        if tool_results:
                            # Add the assistant's tool call message
                            chat_history.add_assistant_message(content or "")
                            
                            # Add the tool responses
                            for result in tool_results:
                                # Use add_tool_message
                                chat_history.add_tool_message(json.dumps(result, ensure_ascii=False))
                            
                            # Recursive call to get the final response
                            return await self.get_chat_message_contents(
                                chat_history, settings, **kwargs
                            )
                    
                    # Create the response message
                    response_message = ChatMessageContent(
                        role=AuthorRole.ASSISTANT,
                        content=content,
                        ai_model_id=self.model_id,
                        metadata={
                            **result,
                            "tool_calls": tool_calls
                        }
                    )
                    
                    return [response_message]
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Ollama request failed: {response.status} - {error_text}")
                    raise Exception(f"Ollama request failed: {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("❌ Ollama request timed out")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama request exception: {e}")
            raise
    
    def _parse_functools_format(self, content: str) -> Tuple[str, List[Dict]]:
        """Parses the functools[...] format for function calls"""
        tool_calls = []
        
        # Find functools[...] formatted function calls
        pattern = r'functools\[(.*?)\]'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Parse the JSON array
                calls = json.loads(f"[{match}]")
                for call in calls:
                    # Convert to the standard tool_call format
                    tool_call = {
                        "function": {
                            "name": call.get("name"),
                            "arguments": call.get("arguments", {})
                        }
                    }
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse function call: {match}")
        
        # Clean the function calls from the content
        cleaned_content = re.sub(pattern, '', content).strip()
        
        return cleaned_content, tool_calls
    
    async def _execute_tool_calls(self, tool_calls: List[Dict], kernel) -> List[Dict]:
        """Executes tool calls"""
        
        results = []
        for call in tool_calls:
            if "function" in call:
                func_info = call["function"]
                func_name = func_info.get("name")
                func_args = func_info.get("arguments", {})
                
                try:
                    logger.info(f"  ✔️  Executing function: {func_name}")
                    logger.info(f"     Arguments: {func_args}")
                    
                    # Find and execute the function
                    result = await self._execute_function(kernel, func_name, func_args)
                    results.append({
                        "name": func_name,
                        "result": f"{str(result)[:100]}..." if len(str(result)) > 100 else f"{result}"
                    })
                    logger.info(f"     ✅ Success: {str(result)[:100]}..." if len(str(result)) > 100 else f"     ✅ Success: {result}")
                except Exception as e:
                    logger.error(f"  ❌ Function execution failed: {e}")
                    logger.debug("     Error details: ", exc_info=True)
                    results.append({
                        "name": func_name,
                        "error": str(e)
                    })
        
        return results
    
    async def _execute_function(self, kernel, func_name: str, func_args: Dict[str, Any]):
        """Executes a single function"""
        from semantic_kernel.functions.kernel_arguments import KernelArguments
        
        # Try to find the function in different plugins
        for plugin_name in ["PlaywrightBrowser", "BrowserAgentSkills"]:
            try:
                result = await kernel.invoke(
                    plugin_name=plugin_name,
                    function_name=func_name,
                    arguments=KernelArguments(**func_args)
                )
                return str(result)
            except Exception:
                continue
        
        raise Exception(f"Function not found: {func_name}")
    
    async def get_chat_message_content(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs
    ) -> ChatMessageContent:
        """
        Gets a single chat message content
        
        Args:
            chat_history: The chat history
            settings: The execution settings
            
        Returns:
            The chat message content
        """
        messages = await self.get_chat_message_contents(chat_history, settings, **kwargs)
        return messages[0] if messages else None
    
    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs
    ):
        """
        Streams a chat completion request
        
        Args:
            chat_history: The chat history
            settings: The execution settings
            
        Yields:
            Streaming chat message content
        """
        await self._ensure_session()
        
        # Convert settings
        options = self._convert_settings(settings)
        
        # Build the request
        messages = self._format_chat_history(chat_history)
        
        # Build the request data
        request_data = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "options": options
        }
        
        # Add function calling support
        tools = kwargs.get("tools", [])
        if tools:
            request_data["tools"] = self._format_tools_for_api(tools)
        
        try:
            async with self._session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if data.get("message"):
                                    content = data["message"].get("content", "")
                                    if content:
                                        yield StreamingChatMessageContent(
                                            role=AuthorRole.ASSISTANT,
                                            content=content,
                                            choice_index=0,
                                            ai_model_id=self.model_id
                                        )
                                
                                # Handle the final message (may contain tool calls)
                                if data.get("done") and data.get("message", {}).get("tool_calls"):
                                    # Streaming response is complete, contains tool calls
                                    # Tool execution logic can be triggered here
                                    pass
                                    
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Ollama streaming request failed: {response.status} - {error_text}")
                    raise Exception(f"Ollama streaming request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Ollama streaming request exception: {e}")
            raise
    
    def _convert_settings(self, settings: PromptExecutionSettings) -> Dict[str, Any]:
        """Converts execution settings to the Ollama format"""
        options = {}
        
        if isinstance(settings, OllamaPromptExecutionSettings):
            if settings.temperature is not None:
                options["temperature"] = settings.temperature
            if settings.max_tokens is not None:
                options["num_predict"] = settings.max_tokens
            if settings.top_p is not None:
                options["top_p"] = settings.top_p
            if settings.top_k is not None:
                options["top_k"] = settings.top_k
            if settings.repeat_penalty is not None:
                options["repeat_penalty"] = settings.repeat_penalty
            if settings.seed is not None:
                options["seed"] = settings.seed
            if settings.stop:
                options["stop"] = settings.stop
            if settings.mirostat is not None:
                options["mirostat"] = settings.mirostat
            if settings.mirostat_tau is not None:
                options["mirostat_tau"] = settings.mirostat_tau
            if settings.mirostat_eta is not None:
                options["mirostat_eta"] = settings.mirostat_eta
            if settings.tfs_z is not None:
                options["tfs_z"] = settings.tfs_z
        else:
            # Use default settings
            options["temperature"] = 0.0
        
        return options
    
    def _format_chat_history(self, chat_history: ChatHistory) -> List[Dict[str, Any]]:
        """Formats chat history for Ollama"""
        messages = []
        
        for message in chat_history.messages:
            role = self._convert_role(message.role)
            msg_dict = {
                "role": role,
                "content": message.content
            }
            
            # If the message contains images (for multimodal models)
            if hasattr(message, "images") and message.images:
                msg_dict["images"] = message.images
            
            messages.append(msg_dict)
        
        return messages
    
    def _convert_role(self, role: AuthorRole) -> str:
        """Converts a role to the Ollama format"""
        role_mapping = {
            AuthorRole.SYSTEM: "system",
            AuthorRole.USER: "user",
            AuthorRole.ASSISTANT: "assistant",
            AuthorRole.TOOL: "tool"  # The official API supports the tool role
        }
        return role_mapping.get(role, "user")
    
    def _format_tools_for_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formats tool definitions for the official API"""
        formatted_tools = []
        
        for tool in tools:
            # If it's already in the correct format, use it directly
            if isinstance(tool, dict) and "type" in tool and tool["type"] == "function":
                formatted_tools.append(tool)
            else:
                # Convert to the correct format
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                formatted_tools.append(formatted_tool)
        
        return formatted_tools
    
    async def close(self):
        """Closes the service"""
        if self._session and not self._session.closed:
            await self._session.close()

class OllamaService:
    """Ollama service wrapper"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initializes the Ollama service
        
        Args:
            config: Ollama configuration
        """
        self.config = config or self._load_config()
        self.chat_service = OllamaChatCompletion(
            model_id=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
    
    def _load_config(self) -> OllamaConfig:
        """Loads configuration from a file or environment variables"""
        # Default configuration
        config = OllamaConfig()
        
        # Try to load from a configuration file
        config_paths = [
            "config/ollama_config.yaml",
            "../config/ollama_config.yaml",
            os.path.join(os.path.dirname(__file__), "../config/ollama_config.yaml")
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                        
                    # Update configuration
                    config.model = yaml_config.get('model', config.model)
                    config.base_url = yaml_config.get('base_url', config.base_url)
                    config.temperature = yaml_config.get('temperature', config.temperature)
                    config.timeout = yaml_config.get('timeout', config.timeout)
                    
                    # Set parameters
                    settings = yaml_config.get('settings', {})
                    config.max_tokens = settings.get('max_tokens', config.max_tokens)
                    config.top_p = settings.get('top_p', config.top_p)
                    config.top_k = settings.get('top_k', config.top_k)
                    
                    logger.info(f"✅ Loaded Ollama configuration from file: {path}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️  Could not load configuration file {path}: {e}")
        
        # Override with environment variables
        if os.environ.get('OLLAMA_MODEL'):
            config.model = os.environ['OLLAMA_MODEL']
            logger.info(f"🔧 Using environment variable OLLAMA_MODEL: {config.model}")
        
        if os.environ.get('OLLAMA_BASE_URL'):
            config.base_url = os.environ['OLLAMA_BASE_URL']
            logger.info(f"🔧 Using environment variable OLLAMA_BASE_URL: {config.base_url}")
        
        return config
    
    async def check_health(self) -> bool:
        """Checks the health of the Ollama service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """Lists the available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    return []
        except Exception:
            return []
    
    async def close(self):
        """Closes the service"""
        await self.chat_service.close()

# Convenience function
def create_ollama_service(
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> OllamaChatCompletion:
    """
    Creates an Ollama chat service that is compatible with the official API
    
    Args:
        model: The model name (if None, read from config file or environment variables)
        base_url: The Ollama API URL (if None, read from config file or environment variables)
        
    Returns:
        OllamaChatCompletion: A chat service instance
    """
    # Create a service instance
    service = OllamaService()
    
    # If parameters are provided, override the configuration
    if model:
        service.config.model = model
        service.chat_service.model_id = model
    
    if base_url:
        service.config.base_url = base_url
        service.chat_service.base_url = base_url
    
    return service.chat_service
