# Microsoft Playwright-MCP + Ollama + Semantic Kernel Project

This project integrates Microsoft's official playwright-mcp with Ollama and Semantic Kernel to build intelligent browser automation applications.

## System Requirements

- macOS (tested on Mac mini Pro)
- Python 3.8+
- Node.js 16+
- Ollama
- 8GB+ RAM (16GB recommended)

## Installation Steps

### 1. Install System Dependencies

```bash
# Install dependencies using Homebrew
brew install node python@3.11

# Install Ollama
brew install ollama
```

### 2. Install playwright-mcp

```bash
# Global installation
npm install -g @playwright/mcp

# Or use npx (recommended)
npx @playwright/mcp@latest
```

### 3. Set up Python Environment

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Ollama Models

```bash
# Start Ollama service
ollama serve

# Pull base model
ollama pull llama3.2

# Create function-calling model (using setup script)
./setup_models.sh
```

## Project Structure

```
playwright-mcp-ollama/
├── semantic-kernel-client/  # Semantic Kernel client
│   ├── main.py              # Main entry point
│   ├── mcp_integration.py   # MCP integration module
│   ├── ollama_service.py    # Ollama service wrapper
│   ├── browser_agent.py     # Browser automation agent
│   └── test_integration.py  # Integration tests
├── config/                  # Configuration files
│   ├── app_config.yaml      # Application configuration
│   └── ollama_config.yaml   # Ollama configuration
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── start_services.sh        # Script to start all services
```

## Usage

### Starting the Services

```bash
# Start all services
./start_services.sh

# Or start them separately
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start playwright-mcp
npx @playwright/mcp@latest

# Terminal 3: Run the main application
python semantic-kernel-client/main.py
```

## Features

- ✅ Based on Microsoft's official playwright-mcp
- ✅ Full support for the MCP protocol
- ✅ Deep integration with Semantic Kernel
- ✅ Uses local models with Ollama
- ✅ Supports function calling (compliant with the official Ollama API)
- ✅ Supports multiple models (llama3.2, phi4-mini, etc.)
- ✅ Accessibility tree mode (default)
- ✅ Vision mode (optional)
- ✅ Complete error handling and retry mechanisms
- ✅ Flexible configuration system

## Configuration

### Ollama Model Configuration

The default model is `llama3.2:function-calling`. You can change it in the following ways:

1. **Configuration File**: Edit `config/ollama_config.yaml`
2. **Environment Variable**: Set the `OLLAMA_MODEL` environment variable
3. **Code Parameter**: Specify the model when creating the service

```python
# Use default configuration
ollama = create_ollama_service()

# Specify a model
ollama = create_ollama_service(model="phi4-mini:function-calling")
```

## Troubleshooting

### playwright-mcp Connection Issues
- Ensure Node.js is installed: `node --version`
- Verify playwright-mcp: `npx @playwright/mcp@latest --version`
- Check for port conflicts: `lsof -i :8931`

### Ollama-related Issues
- Ensure the service is running: `ps aux | grep ollama`
- Check the model list: `ollama list`
- View logs: `tail -f ~/.ollama/logs/server.log`

### Python Environment Issues
- Activate the virtual environment: `source .venv/bin/activate`
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Check the Python version: `python --version`

## License

MIT License
