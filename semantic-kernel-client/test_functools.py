#!/usr/bin/env python3
"""
Test functools format parsing
"""

import pytest
import asyncio
import logging
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from ollama_service import OllamaChatCompletion, OllamaPromptExecutionSettings
from mcp_integration import create_playwright_plugin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

@pytest.mark.asyncio
async def test_functools_parsing():
    print("🧪 Testing functools format parsing")
    print("=" * 60)
    
    # Create a kernel
    kernel = Kernel()
    
    # Create an Ollama service
    ollama = OllamaChatCompletion(model_id="llama3.2:function-calling")
    kernel.add_service(ollama)
    
    # Add the Playwright plugin
    playwright_plugin = await create_playwright_plugin()
    kernel.add_plugin(playwright_plugin)
    
    # Create chat history
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are an intelligent browser automation assistant. You can use various browser tools to complete tasks."
    )
    chat_history.add_user_message("Open Gmail")
    
    # Settings
    settings = OllamaPromptExecutionSettings(temperature=0.0)
    
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
        }
    ]
    
    try:
        # Send the request
        response = await ollama.get_chat_message_content(
            chat_history=chat_history,
            settings=settings,
            kernel=kernel,
            tools=tools
        )
        
        print(f"\n📧 Final response:")
        print(f"{response.content}")
        
        # Check the metadata
        if response.metadata:
            tool_calls = response.metadata.get("tool_calls", [])
            if tool_calls:
                print(f"\n🔧 Detected tool calls:")
                for call in tool_calls:
                    if "function" in call:
                        func = call["function"]
                        print(f"  - {func.get('name')}: {func.get('arguments')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await ollama.close()
        await playwright_plugin.close()

if __name__ == "__main__":
    asyncio.run(test_functools_parsing())
