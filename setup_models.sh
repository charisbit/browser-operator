#!/bin/bash
# Set up Ollama models

echo "🚀 Setting up Ollama function-calling models..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install Ollama first."
    echo "Visit https://ollama.ai to download and install"
    exit 1
fi

# Check if the Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama service is not running. Starting it..."
    ollama serve &
    sleep 5
fi

# Pull the base model
echo "📥 Pulling base models..."
ollama pull llama3.2:3b || echo "⚠️  Could not pull llama3.2"

# Create function-calling optimized models
echo "🔧 Creating function-calling optimized models..."

# Use the standard Modelfile
if [ -f "Modelfile" ]; then
    echo "  - Creating llama3.2:function-calling"
    ollama create llama3.2:function-calling -f Modelfile
fi

# Check if a phi model exists
if ollama list | grep -q "phi"; then
    # Use the phi4-specific Modelfile
    if [ -f "Modelfile.phi4" ]; then
        echo "  - Creating phi4-mini:function-calling"
        ollama create phi4-mini:function-calling -f Modelfile.phi4
    fi
else
    echo "ℹ️  phi model not installed, skipping phi4-mini:function-calling"
fi

# List available models
echo ""
echo "✅ Available models:"
ollama list

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage:"
echo "1. Default model: llama3.2:function-calling"
echo "2. Alternative model: phi4-mini:function-calling (if available)"
echo ""
echo "Configuration file: config/ollama_config.yaml"
echo "You can override the default model with the OLLAMA_MODEL environment variable"
echo ""
echo "Run the demo:"
echo "python demo_ollama.py"
