#!/usr/bin/env python3
"""
Simple test to verify the correct way to invoke prompts in Semantic Kernel
"""

import asyncio
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from ollama_service import OllamaChatCompletion

async def test_prompt_invocation():
    print("Testing prompt invocation methods...")
    
    # Create kernel and add Ollama service
    kernel = Kernel()
    ollama = OllamaChatCompletion(
        model_id="llama3.2:function-calling",
        base_url="http://localhost:11434"
    )
    kernel.add_service(ollama)
    
    # Method 1: Direct chat completion
    print("\nMethod 1: Using chat completion service directly")
    try:
        chat_service = kernel.get_service(type=ChatCompletionClientBase)
        chat_history = ChatHistory()
        chat_history.add_system_message("You are a helpful assistant.")
        chat_history.add_user_message("Say hello in a friendly way.")
        
        settings = PromptExecutionSettings(temperature=0.7)
        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=settings
        )
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Method 1 error: {e}")
    
    # Method 2: Create a prompt function
    print("\nMethod 2: Creating a prompt function")
    try:
        # Create a prompt function
        @kernel_function(
            name="greet",
            description="Generate a greeting"
        )
        async def greet(name: str) -> str:
            prompt = f"Generate a friendly greeting for someone named {name}"
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            chat_service = kernel.get_service(type=ChatCompletionClientBase)
            settings = PromptExecutionSettings(temperature=0.7)
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            return response.content
        
        # Add function to kernel
        kernel.add_function(plugin_name="Greetings", function=greet)
        
        # Invoke the function
        result = await kernel.invoke(
            plugin_name="Greetings",
            function_name="greet",
            arguments=KernelArguments(name="Alice")
        )
        print(f"Function result: {result}")
        
    except Exception as e:
        print(f"Method 2 error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ollama.close()

if __name__ == "__main__":
    asyncio.run(test_prompt_invocation())
