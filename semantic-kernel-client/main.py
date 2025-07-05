"""
Main entry point - Playwright MCP + Ollama + Semantic Kernel
"""
import asyncio
import logging
import sys
from typing import Optional
import colorlog
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from browser_agent import BrowserAgent, create_browser_agent
from ollama_service import OllamaService

# Set up Rich console
console = Console()

# Configure colored logging
def setup_logging():
    """Configures the logging system"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler (colored)
    console_handler = colorlog.StreamHandler()
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

class InteractiveCLI:
    """Interactive command-line interface"""
    
    def __init__(self):
        self.agent: Optional[BrowserAgent] = None
        
    def display_welcome(self):
        """Displays the welcome screen"""
        welcome_text = Text()
        welcome_text.append("🤖 Playwright MCP + Ollama + Semantic Kernel\n", style="bold blue")
        welcome_text.append("═" * 60 + "\n", style="blue")
        welcome_text.append("Intelligent Browser Automation System\n", style="green")
        welcome_text.append("Type 'quit' or 'exit' to exit the program.\n", style="dim")
        
        panel = Panel(welcome_text, title="Welcome", border_style="blue")
        console.print(panel)
    
    async def initialize_agent(self) -> bool:
        """Initializes the browser agent"""
        try:
            console.print("🚀 [yellow]Initializing system...[/yellow]")
            
            ollama_service = OllamaService()
            if not await ollama_service.check_health():
                console.print("❌ [red]Ollama service is not running, please start it first: ollama serve[/red]")
                return False
            
            models = await ollama_service.list_models()
            console.print(f"✅ [green]Ollama connected, available models: {len(models)}[/green]")
            
            model = ollama_service.config.model
            if model not in models and models:
                console.print(f"⚠️  [yellow]Recommended model {model} not found, will use the first available model: {models[0]}[/yellow]")
                model = models[0]
            
            self.agent = await create_browser_agent(ollama_config=ollama_service.config)
            
            console.print("✅ [green]System initialized successfully![/green]")
            console.print(f"🤖 [cyan]Ollama model: {model}[/cyan]")
            
            await ollama_service.close()
            return True
            
        except Exception as e:
            console.print(f"❌ [red]Initialization failed: {e}[/red]")
            logging.error("Initialization failed", exc_info=True)
            return False
    
    async def task_loop(self):
        """Main task loop, continuously accepts user commands"""
        while True:
            console.print("\n" + "─" * 60)
            user_input = Prompt.ask("💬 [cyan]Please enter your task command[/cyan]")

            if user_input.lower() in ["quit", "exit"]:
                break

            if not user_input.strip():
                console.print("⚠️ [yellow]Please enter a valid command[/yellow]")
                continue
            
            console.print(f"\n🔄 [yellow]Executing: {user_input}[/yellow]")
            
            try:
                result = await self.agent.execute_prompt(user_input)
                
                result_panel = Panel(
                    Text(result, overflow="fold"),
                    title="🎯 Execution Result", 
                    border_style="green",
                    expand=True
                )
                console.print(result_panel)
                
            except Exception as e:
                console.print(f"❌ [red]Execution failed: {e}[/red]")
                logging.error(f"Error executing command '{user_input}'", exc_info=True)

    async def run(self):
        """Runs the main program"""
        setup_logging()
        self.display_welcome()
        
        if not await self.initialize_agent():
            console.print("\n❌ [red]System initialization failed, the program will now exit[/red]")
            return
        
        try:
            await self.task_loop()
        except KeyboardInterrupt:
            console.print("\n\n👋 [yellow]Received interrupt signal, exiting...[/yellow]")
        finally:
            if self.agent:
                await self.agent.close()
            console.print("\n👋 [green]Thank you for using, goodbye![/green]")

async def main():
    """Main function"""
    cli = InteractiveCLI()
    await cli.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Program interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Program runtime exception: {e}[/red]")
        sys.exit(1)
