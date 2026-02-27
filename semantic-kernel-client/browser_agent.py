"""
Browser Agent - Intelligent browser automation agent
Combines advanced features of Playwright MCP and Ollama
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from mcp_integration import PlaywrightMCPPlugin, create_playwright_plugin
from ollama_service import OllamaChatCompletion, OllamaConfig

logger = logging.getLogger(__name__)

@dataclass
class BrowserTask:
    """Browser task definition"""
    name: str
    description: str
    steps: List[str]
    expected_output: Optional[str] = None
    
@dataclass
class TaskResult:
    """Task execution result"""
    task_name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration: float = 0.0
    steps_completed: Optional[List[str]] = None

class BrowserAgent:
    """Intelligent browser automation agent"""
    
    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        ollama_config: Optional[OllamaConfig] = None,
        playwright_transport: Optional[Any] = None
    ):
        """
        Initializes the browser agent
        
        Args:
            kernel: Semantic Kernel instance
            ollama_config: Ollama configuration
            playwright_transport: Playwright MCP transport configuration
        """
        self.kernel = kernel or Kernel()
        self.ollama_config = ollama_config or OllamaConfig()
        self.playwright_transport = playwright_transport
        self.playwright_plugin: Optional[PlaywrightMCPPlugin] = None
        self.chat_history = ChatHistory()
        
    async def initialize(self):
        """Initializes the agent"""
        logger.info("🚀 Initializing browser agent...")
        
        # Add Ollama service
        if not self.kernel.services:
            ollama_service = OllamaChatCompletion(
                model_id=self.ollama_config.model,
                base_url=self.ollama_config.base_url,
                timeout=self.ollama_config.timeout
            )
            self.kernel.add_service(ollama_service)
            logger.info("✅ Ollama service added")
        
        # Add Playwright plugin
        if not self.playwright_plugin:
            self.playwright_plugin = await create_playwright_plugin(self.playwright_transport)
            self.kernel.add_plugin(self.playwright_plugin)
            logger.info("✅ Playwright MCP plugin added")
        
        # Add custom skills
        self._register_custom_skills()
        
        logger.info("✅ Browser agent initialization complete")
    
    def _register_custom_skills(self):
        """Registers custom skills"""
        
        @kernel_function(
            name="analyze_page_content",
            description="Analyzes page content and extracts key information"
        )
        async def analyze_page_content(content: str) -> str:
            """Analyzes page content"""
            prompt = f"""
            Analyze the following page content and extract key information:
            
            {content}
            
            Please provide:
            1. Page topic
            2. Key points (3-5)
            3. Important links or resources
            4. A brief summary
            """
            
            # Use chat completion service
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            settings = PromptExecutionSettings(temperature=0.7)
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            return response.content if response else ""
        
        @kernel_function(
            name="create_test_plan",
            description="Creates a test plan for a web page"
        )
        async def create_test_plan(url: str, requirements: str = "") -> str:
            """Creates a web page test plan"""
            prompt = f"""
            Create a detailed test plan for the following web page:
            URL: {url}
            Requirements: {requirements}
            
            Please include:
            1. Functional test cases
            2. UI test cases
            3. Performance testing suggestions
            4. Accessibility checkpoints
            """
            
            # Use chat completion service
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            settings = PromptExecutionSettings(temperature=0.7)
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            return response.content if response else ""
        
        # Register skills with the kernel
        self.kernel.add_function(plugin_name="BrowserAgentSkills", function=analyze_page_content)
        self.kernel.add_function(plugin_name="BrowserAgentSkills", function=create_test_plan)
    
    async def execute_task(self, task: BrowserTask) -> TaskResult:
        """
        Executes a browser task
        
        Args:
            task: Browser task definition
            
        Returns:
            TaskResult: The result of the task execution
        """
        start_time = datetime.now()
        steps_completed = []
        
        try:
            logger.info(f"🎯 Starting task: {task.name}")
            
            # Use AI to understand the task and generate an execution plan
            plan_prompt = f"""
            Task Name: {task.name}
            Task Description: {task.description}
            
            Steps:
            {chr(10).join(f'- {step}' for step in task.steps)}
            
            Please generate a detailed execution plan for this task using the available browser tools.
            """
            
            # Use chat completion service
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
            
            chat_history = ChatHistory()
            chat_history.add_system_message("You are an intelligent browser automation assistant capable of using various browser tools to complete tasks.")
            chat_history.add_user_message(plan_prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            
            # Enable function calling
            settings = PromptExecutionSettings(
                temperature=0.7,
                function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"included_plugins": ["PlaywrightBrowser"]})
            )
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings,
                kernel=self.kernel
            )
            result = response.content if response else ""
            
            # Record completed steps
            steps_completed = task.steps
            
            # Calculate execution time
            duration = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_name=task.name,
                success=True,
                output=str(result),
                duration=duration,
                steps_completed=steps_completed
            )
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_name=task.name,
                success=False,
                output="",
                error=str(e),
                duration=duration,
                steps_completed=steps_completed
            )
    
    async def execute_prompt(self, prompt: str) -> str:
        """
        Executes a natural language prompt
        
        Args:
            prompt: The user's prompt
            
        Returns:
            The execution result
        """
        try:
            logger.info(f"💬 Executing prompt: {prompt}")
            
            # Use chat completion service, enable function calling
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
            
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            
            # Enable automatic function calling, allowing all plugins
            settings = PromptExecutionSettings(
                temperature=0.0,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
            logger.debug(self.playwright_plugin.client.tools)
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings,
                kernel=self.kernel,  # Pass the kernel to enable function calling
                tools=self.playwright_plugin.client.tools  # Get all function metadata
            )
            result = response.content if response else ""
            
            return str(result)
            
        except Exception as e:
            logger.error(f"❌ Prompt execution failed: {e}", exc_info=True)
            return f"Execution failed: {str(e)}"
    
    async def extract_data(self, url: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structured data from a web page
        
        Args:
            url: The URL of the web page
            data_schema: The data schema definition
            
        Returns:
            The extracted data
        """
        try:
            # Navigate to the page
            await self.kernel.invoke(
                plugin_name="PlaywrightBrowser",
                function_name="browser_navigate",
                arguments=KernelArguments(url=url)
            )
            
            # Get a page snapshot
            snapshot = await self.kernel.invoke(
                plugin_name="PlaywrightBrowser",
                function_name="browser_snapshot"
            )
            
            # Use AI to extract data
            extract_prompt = f"""
            Extract data from the following page content according to the specified schema:
            
            Page Content:
            {snapshot}
            
            Data Schema:
            {data_schema}
            
            Please return the data in JSON format that conforms to the schema.
            """
            
            # Use chat completion service
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            
            chat_history = ChatHistory()
            chat_history.add_user_message(extract_prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            settings = PromptExecutionSettings(temperature=0.0)
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            result = response.content if response else ""
            
            # Parse the result
            try:
                import json
                return json.loads(str(result))
            except json.JSONDecodeError:
                return {"error": "Could not parse extracted data", "raw": str(result)}
                
        except Exception as e:
            logger.error(f"❌ Data extraction failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def monitor_page_changes(self, url: str, interval: int = 60, duration: int = 300):
        """
        Monitors a page for changes
        
        Args:
            url: The URL to monitor
            interval: The check interval (in seconds)
            duration: The monitoring duration (in seconds)
        """
        start_time = datetime.now()
        previous_snapshot = None
        changes = []
        
        try:
            while (datetime.now() - start_time).total_seconds() < duration:
                # Navigate to the page
                await self.kernel.invoke(
                    plugin_name="PlaywrightBrowser",
                    function_name="browser_navigate",
                    arguments=KernelArguments(url=url)
                )
                
                # Get a snapshot
                current_snapshot = await self.kernel.invoke(
                    plugin_name="PlaywrightBrowser",
                    function_name="browser_snapshot"
                )
                
                # Compare for changes
                if previous_snapshot and current_snapshot != previous_snapshot:
                    change_time = datetime.now()
                    
                    # Analyze the changes
                    analyze_prompt = f"""
                    Compare the two page snapshots and describe the main changes:
                    
                    Before:
                    {previous_snapshot[:500]}...
                    
                    Now:
                    {current_snapshot[:500]}...
                    
                    Please briefly describe what has changed.
                    """
                    
                    # Use chat completion service
                    from semantic_kernel.contents.chat_history import ChatHistory
                    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
                    from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
                    
                    chat_history = ChatHistory()
                    chat_history.add_user_message(analyze_prompt)
                    
                    chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
                    settings = PromptExecutionSettings(temperature=0.5)
                    
                    response = await chat_service.get_chat_message_content(
                        chat_history=chat_history,
                        settings=settings
                    )
                    change_description = response.content if response else "Could not analyze changes"
                    
                    changes.append({
                        "time": change_time.isoformat(),
                        "description": str(change_description)
                    })
                    
                    logger.info(f"📝 Change detected: {change_description}")
                
                previous_snapshot = current_snapshot
                
                # Wait for the next check
                await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
        
        return changes
    
    async def run_accessibility_audit(self, url: str) -> Dict[str, Any]:
        """
        Runs an accessibility audit
        
        Args:
            url: The URL to audit
            
        Returns:
            The audit results
        """
        try:
            # Navigate to the page
            await self.kernel.invoke(
                plugin_name="PlaywrightBrowser",
                function_name="browser_navigate",
                arguments=KernelArguments(url=url)
            )
            
            # Get a page snapshot (including accessibility tree)
            snapshot = await self.kernel.invoke(
                plugin_name="PlaywrightBrowser",
                function_name="browser_snapshot"
            )
            
            # Analyze accessibility
            audit_prompt = f"""
            Perform an accessibility audit on the following page:
            
            {snapshot}
            
            Please check for:
            1. Images with alt text
            2. Form elements with labels
            3. Correct heading hierarchy
            4. Color contrast issues
            5. Keyboard navigation support
            6. Use of ARIA attributes
            
            For each issue, provide:
            - A description of the issue
            - The severity (high/medium/low)
            - A suggested fix
            """
            
            # Use chat completion service
            from semantic_kernel.contents.chat_history import ChatHistory
            from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
            from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
            
            chat_history = ChatHistory()
            chat_history.add_user_message(audit_prompt)
            
            chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
            settings = PromptExecutionSettings(temperature=0.0)
            
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            result = response.content if response else "Could not complete audit"
            
            return {
                "url": url,
                "audit_time": datetime.now().isoformat(),
                "findings": str(result)
            }
            
        except Exception as e:
            logger.error(f"❌ Audit failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Closes the agent"""
        if self.playwright_plugin:
            await self.playwright_plugin.close()
        
        # Close the Ollama service
        for service in self.kernel.services.values():
            if hasattr(service, 'close'):
                await service.close()

# Convenience function
async def create_browser_agent(
    ollama_config: Optional[OllamaConfig] = None,
    playwright_transport: Optional[Any] = None
) -> BrowserAgent:
    """
    Creates and initializes a browser agent
    
    Args:
        ollama_config: Ollama configuration
        playwright_transport: Playwright transport configuration
        
    Returns:
        BrowserAgent: An initialized agent
    """
    config = ollama_config or OllamaConfig()
    agent = BrowserAgent(ollama_config=config, playwright_transport=playwright_transport)
    await agent.initialize()
    return agent
