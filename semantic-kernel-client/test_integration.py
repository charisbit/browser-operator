"""
Integration test script
Tests the full functionality of Playwright MCP + Ollama + Semantic Kernel
"""
import pytest
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'semantic-kernel-client'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mcp_integration import PlaywrightMCPClient
from ollama_service import OllamaService
from browser_agent import create_browser_agent

console = Console()

class IntegrationTester:
    """Integration tester"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, test_name: str, success: bool, details: str = ""):
        """Adds a test result"""
        self.results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    async def test_ollama_connection(self):
        """Tests the Ollama connection"""
        console.print("\n🧪 Testing Ollama connection...")
        try:
            service = OllamaService()
            if await service.check_health():
                models = await service.list_models()
                self.add_result(
                    "Ollama Connection",
                    True,
                    f"Found {len(models)} models"
                )
                console.print("✅ Ollama connection successful")
            else:
                self.add_result("Ollama Connection", False, "Service not running")
                console.print("❌ Ollama connection failed")
            await service.close()
        except Exception as e:
            self.add_result("Ollama Connection", False, str(e))
            console.print(f"❌ Ollama test exception: {e}")
    
    async def test_playwright_mcp(self):
        """Tests Playwright MCP"""
        console.print("\n🧪 Testing Playwright MCP...")
        try:
            client = PlaywrightMCPClient()
            if await client.connect():
                tools_count = len(client.tools)
                self.add_result(
                    "Playwright MCP",
                    True,
                    f"Found {tools_count} tools"
                )
                console.print(f"✅ Playwright MCP connection successful, number of tools: {tools_count}")
                
                # Test a simple tool call
                result = await client.call_tool("browser_navigate", {"url": "https://example.com"})
                if result.get("success"):
                    self.add_result("Tool Call Test", True, "browser_navigate successful")
                    console.print("✅ Tool call test passed")
                else:
                    self.add_result("Tool Call Test", False, result.get("error", "Unknown error"))
                    console.print("❌ Tool call test failed")
                
                await client.close()
            else:
                self.add_result("Playwright MCP", False, "Connection failed")
                console.print("❌ Playwright MCP connection failed")
        except Exception as e:
            self.add_result("Playwright MCP", False, str(e))
            console.print(f"❌ Playwright MCP test exception: {e}")
    
    async def test_browser_agent(self):
        """Tests the Browser Agent"""
        console.print("\n🧪 Testing Browser Agent...")
        try:
            agent = await create_browser_agent()
            self.add_result("Browser Agent Initialization", True, "Agent created successfully")
            console.print("✅ Browser Agent initialized successfully")
            
            # Test a simple task
            console.print("   Executing test task...")
            result = await agent.execute_prompt("Get the current page title")
            
            if result and "Error" not in result and "Failed" not in result:
                self.add_result("Task Execution Test", True, "Prompt executed successfully")
                console.print("✅ Task execution test passed")
            else:
                self.add_result("Task Execution Test", False, result[:50] if result else "No result")
                console.print("❌ Task execution test failed")
            
            await agent.close()
        except Exception as e:
            self.add_result("Browser Agent", False, str(e))
            console.print(f"❌ Browser Agent test exception: {e}")
    
    async def test_data_extraction(self):
        """Tests the data extraction functionality"""
        console.print("\n🧪 Testing data extraction...")
        try:
            agent = await create_browser_agent()
            
            # Test extracting data from an example page
            schema = {
                "title": "string",
                "heading": "string",
                "paragraphs": "array"
            }
            
            result = await agent.extract_data("https://google.com", schema)
            
            if result and "error" not in result:
                self.add_result("Data Extraction", True, "Successfully extracted structured data")
                console.print("✅ Data extraction test passed")
            else:
                self.add_result("Data Extraction", False, result.get("error", "Extraction failed"))
                console.print("❌ Data extraction test failed")
                console.print(result["raw"])
            
            await agent.close()
        except Exception as e:
            self.add_result("Data Extraction", False, str(e))
            console.print(f"❌ Data extraction test exception: {e}")
    
    def display_results(self):
        """Displays the test results"""
        console.print("\n" + "="*60)
        
        # Create a results table
        table = Table(title="📊 Integration Test Results", show_header=True, header_style="bold blue")
        table.add_column("Test Item", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Details", style="yellow")
        
        passed = 0
        failed = 0
        
        for result in self.results:
            status = "✅ Passed" if result["success"] else "❌ Failed"
            style = "green" if result["success"] else "red"
            table.add_row(
                result["test"],
                f"[{style}]{status}[/{style}]",
                result["details"]
            )
            
            if result["success"]:
                passed += 1
            else:
                failed += 1
        
        console.print(table)
        
        # Summary
        total = passed + failed
        success_rate = (total > 0) and (passed / total * 100) or 0
        
        summary = f"""
Total tests: {total}
Passed: {passed}
Failed: {failed}
Success rate: {success_rate:.1f}%
"""
        
        if failed == 0:
            summary_panel = Panel(summary, title="✅ All tests passed", border_style="green")
        else:
            summary_panel = Panel(summary, title="⚠️ Some tests failed", border_style="yellow")
        
        console.print(summary_panel)
    
    async def run_all_tests(self):
        """Runs all tests"""
        console.print("🚀 Starting integration tests...")
        console.print("="*60)
        
        # Basic connection tests
        await self.test_ollama_connection()
        await self.test_playwright_mcp()
        
        # Functional tests
        await self.test_browser_agent()
        await self.test_data_extraction()
        
        # Display results
        self.display_results()

async def main():
    """Main function"""
    tester = IntegrationTester()
    
    try:
        await tester.run_all_tests()
        
        # Decide the exit code based on the test results
        failed_count = sum(1 for r in tester.results if not r["success"])
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        console.print("\n⏹️ Tests interrupted")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
