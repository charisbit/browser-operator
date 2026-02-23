#!/usr/bin/env python3
"""
Quick test to verify Ollama connection and basic functionality
"""

import pytest
import asyncio
import logging
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from ollama_service import create_ollama_service

logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_ollama():
    print("Testing Ollama connection...")
    
    # Create kernel and add Ollama service
    kernel = Kernel()
    
    # 使用便捷函数创建服务（自动从配置文件加载设置）
    ollama = create_ollama_service()
    kernel.add_service(ollama)
    
    # Create chat history
    chat_history = ChatHistory()
    chat_history.add_user_message("Hello! Can you say 'Hi' back?")
    
    # Create settings
    settings = PromptExecutionSettings()
    
    try:
        # Test the service
        response = await ollama.get_chat_message_content(
            chat_history=chat_history,
            settings=settings
        )
        
        print(f"✅ Success! Response: {response.content}")
        print(f"📋 Model: {ollama.model_id}")
        print(f"🔗 Base URL: {ollama.base_url}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await ollama.close() 

if __name__ == "__main__":
    asyncio.run(test_ollama())
