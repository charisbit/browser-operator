"""
Test function calling functionality
"""
import asyncio
import logging
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from ollama_service import (
    OllamaConfig, 
    OllamaChatCompletion,
    OllamaPromptExecutionSettings
)
from mcp_integration import create_playwright_plugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_function_calling(model: str = "llama3.2:function-calling"):
    """Tests the function calling capability"""
    
    # Create a kernel
    kernel = Kernel()
    
    # Configure the Ollama service
    config = OllamaConfig(
        model=model,
        temperature=0.0
    )
    
    # Create the chat service
    chat_service = OllamaChatCompletion(
        model_id=config.model,
        base_url=config.base_url
    )
    
    # Add the service to the kernel
    kernel.add_service(chat_service)
    
    # Add the Playwright plugin
    playwright_plugin = await create_playwright_plugin()
    kernel.add_plugin(playwright_plugin)
    
    # Test cases
    test_cases = [
        {
            "name": "Navigation Test",
            "prompt": "Please navigate to https://example.com",
            "expected_function": "browser_navigate"
        },
        {
            "name": "Screenshot Test",
            "prompt": "Please take a screenshot of the current page",
            "expected_function": "browser_take_screenshot"
        },
        {
            "name": "Complex Task Test",
            "prompt": "Please visit https://www.google.com, type 'Semantic Kernel' in the search box, and then click the search button",
            "expected_functions": ["browser_navigate", "browser_type", "browser_click"]
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"Model: {model}")
        print(f"Prompt: {test['prompt']}")
        print(f"{'='*60}")
        
        # Create chat history
        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a browser automation assistant that can use various browser tools to complete tasks."
        )
        chat_history.add_user_message(test['prompt'])
        
        # Execution settings
        settings = OllamaPromptExecutionSettings(
            temperature=0.0,
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )
        
        try:
            # Define tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "description": "Navigate to a URL",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The URL to navigate to"
                                }
                            },
                            "required": ["url"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_take_screenshot",
                        "description": "Take a screenshot of the current page",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "File name to save the screenshot"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_type",
                        "description": "Type text into an element",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "element": {
                                    "type": "string",
                                    "description": "Element description"
                                },
                                "ref": {
                                    "type": "string",
                                    "description": "Element reference"
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Text to type"
                                }
                            },
                            "required": ["element", "ref", "text"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_click",
                        "description": "Click on an element",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "element": {
                                    "type": "string",
                                    "description": "Element description"
                                },
                                "ref": {
                                    "type": "string",
                                    "description": "Element reference"
                                }
                            },
                            "required": ["element", "ref"]
                        }
                    }
                }
            ]
            
            # Send the request
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings,
                kernel=kernel,
                tools=tools
            )
            
            print(f"\nResponse: {response.content}")
            
            # Check for function calls in the metadata
            if response.metadata and "tool_calls" in response.metadata:
                tool_calls = response.metadata["tool_calls"]
                if tool_calls:
                    print(f"\nDetected function calls:")
                    for call in tool_calls:
                        if "function" in call:
                            func = call["function"]
                            print(f"  - {func.get('name')}: {func.get('arguments')}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
    
    # Close the services
    await chat_service.close()
    await playwright_plugin.close()

async def test_raw_api_call():
    """Tests a raw API call"""
    
    print("\n" + "="*60)
    print("Testing raw API call")
    print("="*60)
    
    # Create a simple test service
    chat_service = OllamaChatCompletion(
        model_id="llama3.2:function-calling"
    )
    
    # Test message
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful assistant that can use browser tools."
    )
    chat_history.add_user_message(
        "Navigate to https://example.com"
    )
    
    # Send the request
    settings = OllamaPromptExecutionSettings(temperature=0.0)
    
    try:
        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=settings,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "description": "Navigate to a URL",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to navigate to"}
                            },
                            "required": ["url"]
                        }
                    }
                }
            ]
        )
        
        print(f"Raw response: {response.content}")
        if response.metadata:
            print(f"Metadata: {response.metadata}")
        
    except Exception as e:
        logger.error(f"Raw API test failed: {e}")
    
    await chat_service.close()

async def main():
    """Main test function"""
    
    print("🧪 Starting function calling tests")
    
    # Test raw API call
    await test_raw_api_call()
    
    # Test llama3.2:function-calling
    print("\n\n🦙 Testing llama3.2:function-calling")
    await test_function_calling("llama3.2:function-calling")
    
    # Test phi4-mini:function-calling (if available)
    print("\n\n🤖 Testing phi4-mini:function-calling")
    await test_function_calling("phi4-mini:function-calling")
    
    print("\n✅ Tests complete")

if __name__ == "__main__":
    asyncio.run(main())
