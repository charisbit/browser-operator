#!/bin/bash
#
# Start all services
#

echo "🚀 Starting Playwright MCP + Ollama + Semantic Kernel system"
echo "=================================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

cd "$PROJECT_ROOT"

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "⚠️  Ollama is not running, starting it..."
    ollama serve &
    OLLAMA_PID=$!
    echo "   Waiting for Ollama to start..."
    sleep 5
    
    # Verify that Ollama has started successfully
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "✅ Ollama has started"
    else
        echo "❌ Ollama failed to start"
        echo "Please run manually: ollama serve"
        exit 1
    fi
else
    echo "✅ Ollama is already running"
fi

# Check the models
echo ""
echo "🔍 Checking Ollama models..."
if ollama list | grep -q "llama3.2:function-calling"; then
    echo "✅ Function-calling model is installed"
else
    echo "⚠️  Function-calling model not found"
    echo "   It is recommended to create it: ollama create llama3.2:function-calling -f ../Modelfile"
fi

# Activate the virtual environment
echo ""
echo "🐍 Activating Python virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment does not exist"
    echo "Please run first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Start the main program
echo ""
echo "🎯 Starting main program..."
echo "=================================================="
echo ""

cd semantic-kernel-client
python main.py

# Clean up
if [ ! -z "$OLLAMA_PID" ]; then
    echo ""
    echo "🛑 Stopping Ollama service..."
    kill $OLLAMA_PID 2>/dev/null
fi

_response = ""
echo "👋 System has exited"
