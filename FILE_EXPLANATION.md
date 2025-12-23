# File Structure and Explanation

This document explains each file in the Mainframe Agent project and its purpose.

## Core Agent Files

### 1. `mainframe_agent.py` (844 lines)
**Purpose**: Main agent class that orchestrates automation execution using LangGraph.

**Key Components**:
- **MainframeAgent Class**: Core agent that manages the entire workflow
- **AgentState TypedDict**: Defines the state structure for LangGraph workflow
- **LangGraph Workflow**: Orchestrates step execution with error recovery

**Key Methods**:
- `__init__()`: Initializes agent with configuration
- `_build_workflow()`: Creates LangGraph workflow with nodes and edges
- `execute()`: Main execution method that runs the workflow
- `execute_from_file()`: Loads and executes from JSON/CKP file
- `_prepare_step()`: Prepares step for execution
- `_apply_rules()`: Applies business rules to steps
- `_execute_step()`: Executes step via MCP client
- `_validate_step()`: Validates step execution result
- `_check_exceptions()`: Detects exceptions from screen text
- `_execute_recovery()`: Executes recovery flows for exceptions
- `_handle_error()`: Handles errors using OLLAMA
- `_apply_fix()`: Applies OLLAMA-suggested fixes

**Workflow Nodes**:
1. `prepare_step` → Prepares current step
2. `apply_rules` → Applies business rules
3. `execute_step` → Executes via MCP
4. `validate_step` → Validates result
5. `check_exceptions` → Detects exceptions
6. `execute_recovery` → Runs recovery flows
7. `handle_error` → Uses OLLAMA for fixes
8. `apply_fix` → Applies fixes
9. `make_decision` → Decides next action

---

### 2. `config.py` (89 lines)
**Purpose**: Configuration management using Pydantic settings.

**Key Components**:
- **AgentConfig Class**: Configuration class with all settings
- Environment variable support via `.env` file
- Type validation and defaults

**Configuration Options**:
- `ollama_url`: OLLAMA server URL (default: http://localhost:11434)
- `ollama_model`: OLLAMA model name (default: llama3)
- `mcp_server_url`: MCP Server URL (default: http://localhost:8000)
- `mcp_timeout`: MCP request timeout (default: 30 seconds)
- `max_retries`: Maximum retries per step (default: 3)
- `retry_delay`: Delay between retries (default: 1.0 seconds)
- `log_level`: Logging level (default: INFO)

**Usage**:
```python
from config import AgentConfig

config = AgentConfig(
    ollama_url="http://custom-ollama:11434",
    mcp_server_url="http://custom-mcp:8000"
)
```

---

### 3. `mcp_client.py` (137 lines)
**Purpose**: HTTP client for communicating with MCP (Model Context Protocol) Server.

**Key Components**:
- **MCPClient Class**: Async HTTP client for MCP Server
- Uses `httpx` for async HTTP requests

**Key Methods**:
- `__init__()`: Initializes HTTP client with base URL
- `call_tool()`: Calls a specific MCP tool
- `execute_step()`: Executes a step by calling appropriate MCP tool
- `close()`: Closes HTTP client

**MCP Server Endpoint**:
- **POST** `/tools/{tool_name}`
- Request body: `{"tool": "tool_name", "arguments": {...}}`
- Response: `{"success": true, "result": {...}}`

**Error Handling**:
- HTTP errors → Returns error dict with status code
- Request errors → Returns connection error
- All errors are logged and returned in structured format

---

### 4. `ollama_client.py` (211 lines)
**Purpose**: Client for OLLAMA LLM server for error recovery.

**Key Components**:
- **OllamaClient Class**: Async HTTP client for OLLAMA
- Error analysis and fix generation

**Key Methods**:
- `__init__()`: Initializes OLLAMA client
- `generate_fix()`: Requests fix suggestion from OLLAMA
- `_build_fix_prompt()`: Builds prompt for OLLAMA
- `_parse_fix_suggestion()`: Parses OLLAMA response and updates step

**OLLAMA Endpoint**:
- **POST** `/api/generate`
- Request: `{"model": "llama3", "prompt": "...", "stream": false}`
- Response: `{"response": "fix suggestion..."}`

**Fix Generation Process**:
1. Build prompt with step, error, and context
2. Send to OLLAMA
3. Parse JSON response
4. Extract updated arguments
5. Return fixed step

---

### 5. `ckp_reader.py` (218 lines)
**Purpose**: Reader for CKP (Checkpoint) JSON format files with advanced features.

**Key Components**:
- **CKPReader Class**: Parses and processes CKP files
- Variable substitution
- Schema validation

**Key Methods**:
- `__init__()`: Loads CKP file and variables
- `get_steps()`: Returns steps with variable substitution
- `get_rules()`: Returns business rules
- `get_validations()`: Returns validation definitions
- `get_exceptions()`: Returns exception definitions
- `get_recovery_flows()`: Returns recovery flow definitions
- `validate_variables()`: Validates variables against schema
- `_substitute_variables()`: Replaces `{{variable}}` placeholders

**CKP Format Support**:
- Variable substitution: `{{variable_name}}`
- Business rules with conditions
- Validations with expected text
- Exception detection patterns
- Recovery flow execution

---

### 6. `step_reader.py` (110 lines)
**Purpose**: Simple JSON step file reader (backward compatibility).

**Key Components**:
- **StepReader Class**: Static methods for reading simple JSON
- Supports multiple JSON structures

**Key Methods**:
- `load_steps()`: Loads steps from JSON file
- `validate_step()`: Validates step structure

**Supported Formats**:
1. Direct array: `[{"tool": "...", ...}, ...]`
2. Object with steps: `{"steps": [...]}`
3. Object with workflow: `{"workflow": {"steps": [...]}}`

---

### 7. `main.py` (386 lines)
**Purpose**: Command-line entry point for the agent.

**Key Components**:
- Argument parsing
- File loading and validation
- Execution plan display
- Agent execution

**Command-Line Arguments**:
- `steps_file`: Path to JSON/CKP file (required)
- `--var KEY=VALUE`: Variables (multiple allowed)
- `--vars-file PATH`: Variables JSON file
- `--ollama-url URL`: OLLAMA server URL
- `--ollama-model MODEL`: OLLAMA model name
- `--mcp-url URL`: MCP Server URL
- `--max-retries N`: Maximum retries
- `--log-level LEVEL`: Logging level
- `--skip-preview`: Skip execution plan
- `--no-delay`: Skip delay before execution

**Features**:
- Automatic CKP vs Simple JSON detection
- Variable validation
- Execution plan display
- Detailed result reporting

---

## Supporting Files

### 8. `__init__.py`
**Purpose**: Package initialization, exports main classes.

**Exports**:
- `MainframeAgent`
- `AgentConfig`
- `MCPClient`
- `OllamaClient`
- `StepReader`
- `CKPReader`

---

### 9. `requirements.txt`
**Purpose**: Python package dependencies.

**Dependencies**:
- `langgraph>=0.2.0`: Workflow orchestration
- `langchain-core>=0.3.0`: LangChain core
- `httpx>=0.24.0,<0.28.0`: Async HTTP client
- `pydantic>=2.0.0`: Data validation
- `pydantic-settings>=2.0.0`: Settings management
- `python-dotenv>=1.0.0`: Environment variables

---

### 10. `ckp.json`
**Purpose**: Example CKP format file with full features.

**Contains**:
- Procedure metadata
- Base steps (7 steps)
- Variables schema (required/optional)
- Business rules (3 rules)
- Validations (1 validation)
- Exceptions (2 exceptions)
- Recovery flows (2 recovery flows)
- Provenance metadata

---

### 11. `example_steps.json`
**Purpose**: Simple JSON format example.

**Contains**:
- Simple step array format
- Basic tool/arguments structure

---

## Documentation Files

### 12. `README.md`
Complete project documentation with:
- Features overview
- Installation instructions
- Configuration guide
- Usage examples
- Architecture diagram

### 13. `CKP_FORMAT.md`
CKP format documentation:
- Format structure
- Variable substitution
- Business rules
- Validations
- Exceptions
- Recovery flows

### 14. `QUICKSTART.md`
Quick start guide for new users.

### 15. `SETUP.md`
Setup instructions with dependency conflict resolution.

### 16. `USAGE_EXAMPLE.md`
Code examples for programmatic usage.

---

## Data Flow

```
main.py
  ↓
Loads CKP/JSON file
  ↓
CKPReader/StepReader
  ↓
MainframeAgent
  ↓
LangGraph Workflow
  ├─→ MCPClient → MCP Server (step execution)
  ├─→ OllamaClient → OLLAMA (error recovery)
  └─→ CKPReader (rules, validations, exceptions, recovery)
  ↓
Execution Results
```

## Key Concepts

### LangGraph Workflow
- State-based workflow orchestration
- Conditional routing based on execution results
- Error recovery and retry logic

### MCP Server Integration
- Executes automation steps
- Returns execution results
- Handles tool-specific errors

### OLLAMA Integration
- Analyzes failed steps
- Suggests fixes
- Updates step parameters

### CKP Format
- Advanced automation format
- Business rules and validations
- Exception handling
- Recovery flows

---

## Usage Patterns

### 1. Simple Execution
```python
from mainframe_agent import MainframeAgent

agent = MainframeAgent()
result = await agent.execute_from_file("steps.json")
```

### 2. With Configuration
```python
from mainframe_agent import MainframeAgent
from config import AgentConfig

config = AgentConfig(mcp_server_url="http://custom:8000")
agent = MainframeAgent(config)
result = await agent.execute(steps)
```

### 3. CKP Format
```python
from mainframe_agent import MainframeAgent
from ckp_reader import CKPReader

ckp_reader = CKPReader("ckp.json", {"account_number": "123"})
steps = ckp_reader.get_steps()
agent = MainframeAgent()
result = await agent.execute(steps, ckp_reader=ckp_reader)
```

---

## Error Handling Flow

1. **Step Execution Fails**
   ↓
2. **Check for Exceptions** (CKP format)
   ↓
3. **Execute Recovery Flow** (if exception detected)
   ↓
4. **If Recovery Fails or No Exception**
   ↓
5. **Request Fix from OLLAMA**
   ↓
6. **Apply Fix and Retry** (up to max_retries)
   ↓
7. **Skip or Stop** (based on on_error setting)

---

## Configuration Priority

1. Command-line arguments (highest)
2. Environment variables (.env file)
3. Default values (lowest)

---

This architecture provides a robust, extensible automation framework with intelligent error recovery and advanced workflow orchestration.
