#!/usr/bin/env python3
"""
Quickly verify the functools format fix
"""

import pytest
import asyncio
from ollama_service import OllamaChatCompletion

@pytest.mark.asyncio
async def test_parse():
    service = OllamaChatCompletion()
    
    # Test response content
    test_content = 'functools[{"name": "browser_navigate", "arguments": {"url": "https://mail.google.com/"}}] \n\nPlease note that the browser will be navigated to the Gmail website.'
    
    # Parse
    cleaned, calls = service._parse_functools_format(test_content)
    
    print("Original content:")
    print(test_content)
    print("\n" + "="*60 + "\n")
    
    print("Cleaned content:")
    print(cleaned)
    print("\n" + "="*60 + "\n")
    
    print("Parsed function calls:")
    for call in calls:
        print(call)

if __name__ == "__main__":
    asyncio.run(test_parse())
