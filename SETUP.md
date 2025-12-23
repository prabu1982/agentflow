# Setup Instructions

## Option 1: Virtual Environment (Recommended)

To avoid dependency conflicts with other packages, use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Option 2: Use Minimal Requirements

If you have many packages installed and want to minimize conflicts:

```bash
pip install -r requirements-minimal.txt
```

## Option 3: Install with Conflict Resolution

If you want to install despite conflicts (may cause issues with other packages):

```bash
pip install -r requirements.txt --no-deps
# Then manually install dependencies
pip install langgraph langchain-google-genai langchain-core google-generativeai httpx pydantic pydantic-settings python-dotenv
```

## Dependency Conflicts

The following packages may conflict with existing installations:

- **httpx**: Some packages require `<0.28`, we use `>=0.24.0,<0.28.0`
- **python-dotenv**: Some packages require `<=0.21.1`, we use compatible version
- **langchain packages**: Version conflicts with langflow, langchain-experimental

### Recommended Solution

**Use a virtual environment** to isolate dependencies:

```bash
cd /Users/prabu_k/Documents/POC/AutomationAgent-support
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This ensures the Mainframe Agent has its own isolated environment without conflicts.

## Verify Installation

After installation, verify:

```bash
python -c "from mainframe_agent import MainframeAgent; print('✓ Installation successful')"
```

## Running the Agent

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the agent
python main.py example_steps.json
```
