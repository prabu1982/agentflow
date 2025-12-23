# Usage Examples

## Basic Example

```python
import asyncio
from mainframe_agent import MainframeAgent
from config import AgentConfig

async def main():
    # Create configuration
    config = AgentConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3",
        mcp_server_url="http://localhost:8000",
        max_retries=3
    )
    
    # Create agent
    async with MainframeAgent(config) as agent:
        # Execute from file
        result = await agent.execute_from_file("example_steps.json")
        
        print(f"Success: {result['success']}")
        print(f"Successful steps: {result['successful_steps']}/{result['total_steps']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Programmatic Usage

```python
import asyncio
from mainframe_agent import MainframeAgent
from config import AgentConfig

async def main():
    config = AgentConfig()
    
    # Define steps programmatically
    steps = [
        {
            "tool": "connect_mainframe",
            "arguments": {
                "host": "mainframe.example.com",
                "port": 23
            },
            "description": "Connect to mainframe",
            "on_error": "retry"
        },
        {
            "tool": "send_command",
            "arguments": {
                "command": "DISPLAY TIME"
            },
            "description": "Display time",
            "on_error": "skip"
        }
    ]
    
    async with MainframeAgent(config) as agent:
        result = await agent.execute(steps)
        
        # Process results
        for step_result in result["results"]:
            if step_result["success"]:
                print(f"✓ {step_result['step']['tool']}: Success")
            else:
                print(f"✗ {step_result['step']['tool']}: {step_result['message']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Custom Configuration

```python
from config import AgentConfig

# Override specific settings
config = AgentConfig(
    ollama_url="http://remote-ollama:11434",
    ollama_model="llama3.2",
    mcp_server_url="http://mcp-server:8000",
    max_retries=5,
    retry_delay=2.0,
    mcp_timeout=60
)
```

## Error Handling

The agent automatically handles errors:

1. **Step fails** → Error captured
2. **OLLAMA analyzes error** → Suggests fix
3. **Fix applied** → Step retried
4. **Max retries reached** → Skip or stop based on `on_error`

```json
{
  "tool": "send_command",
  "arguments": {
    "command": "INVALID_COMMAND"
  },
  "on_error": "retry"  // Will retry with LLM fix
}
```

## Command Line Usage

```bash
# Basic usage
python main.py example_steps.json

# With custom configuration
python main.py example_steps.json \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3 \
  --mcp-url http://localhost:8000 \
  --max-retries 5 \
  --log-level DEBUG
```
