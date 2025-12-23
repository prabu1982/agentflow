"""FastAPI server for Mainframe Agent API."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import AgentConfig
from mainframe_agent import MainframeAgent
from ckp_reader import CKPReader
from step_reader import StepReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mainframe Agent API",
    description="REST API for Mainframe Agent automation execution",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (can be reused)
_agent_instance: Optional[MainframeAgent] = None


# Request/Response Models
class ExecuteStepsRequest(BaseModel):
    """Request model for executing steps."""
    steps: List[Dict[str, Any]] = Field(..., description="List of automation steps")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional agent configuration")


class ExecuteFileRequest(BaseModel):
    """Request model for executing from file."""
    file_path: str = Field(..., description="Path to JSON/CKP file")
    variables: Optional[Dict[str, Any]] = Field(None, description="Variables for CKP files")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional agent configuration")


class ExecuteCKPRequest(BaseModel):
    """Request model for executing CKP file."""
    file_path: str = Field(..., description="Path to CKP JSON file")
    variables: Dict[str, Any] = Field(..., description="Variables for variable substitution")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional agent configuration")


class ExecutionResponse(BaseModel):
    """Response model for execution results."""
    success: bool
    total_steps: int
    successful_steps: int
    failed_steps: int
    results: List[Dict[str, Any]]
    execution_time: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    config: Dict[str, Any]


class ExecutionPlanResponse(BaseModel):
    """Response model for execution plan."""
    procedure_id: Optional[str] = None
    description: Optional[str] = None
    total_steps: int
    variables: Dict[str, Any]
    steps: List[Dict[str, Any]]
    rules: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    recovery_flows: List[Dict[str, Any]] = []


def get_agent(config: Optional[Dict[str, Any]] = None) -> MainframeAgent:
    """Get or create agent instance."""
    global _agent_instance
    
    if config:
        agent_config = AgentConfig(**config)
        return MainframeAgent(agent_config)
    
    if _agent_instance is None:
        _agent_instance = MainframeAgent()
    
    return _agent_instance


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": "Mainframe Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "execute_steps": "/api/v1/execute/steps",
            "execute_file": "/api/v1/execute/file",
            "execute_ckp": "/api/v1/execute/ckp",
            "plan": "/api/v1/plan"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        config = AgentConfig()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            config={
                "ollama_url": config.ollama_url,
                "ollama_model": config.ollama_model,
                "mcp_server_url": config.mcp_server_url,
                "max_retries": config.max_retries
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/v1/execute/steps", response_model=ExecutionResponse, tags=["Execution"])
async def execute_steps(request: ExecuteStepsRequest):
    """
    Execute automation steps directly.
    
    Request body:
    ```json
    {
        "steps": [
            {
                "tool": "send_keys",
                "arguments": {
                    "action": "send_keys",
                    "value": "test"
                }
            }
        ],
        "config": {
            "mcp_server_url": "http://localhost:8000",
            "max_retries": 3
        }
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        agent = get_agent(request.config)
        
        result = await agent.execute(request.steps)
        
        execution_time = time.time() - start_time
        
        return ExecutionResponse(
            success=result.get("success", False),
            total_steps=result.get("total_steps", 0),
            successful_steps=result.get("successful_steps", 0),
            failed_steps=result.get("failed_steps", 0),
            results=result.get("results", []),
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/api/v1/execute/file", response_model=ExecutionResponse, tags=["Execution"])
async def execute_file(request: ExecuteFileRequest):
    """
    Execute automation steps from a JSON/CKP file.
    
    Request body:
    ```json
    {
        "file_path": "/path/to/steps.json",
        "variables": {
            "account_number": "123456789"
        },
        "config": {
            "mcp_server_url": "http://localhost:8000"
        }
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        agent = get_agent(request.config)
        
        result = await agent.execute_from_file(
            request.file_path,
            variables=request.variables
        )
        
        execution_time = time.time() - start_time
        
        return ExecutionResponse(
            success=result.get("success", False),
            total_steps=result.get("total_steps", 0),
            successful_steps=result.get("successful_steps", 0),
            failed_steps=result.get("failed_steps", 0),
            results=result.get("results", []),
            execution_time=execution_time
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/api/v1/execute/ckp", response_model=ExecutionResponse, tags=["Execution"])
async def execute_ckp(request: ExecuteCKPRequest):
    """
    Execute CKP format file with variables.
    
    Request body:
    ```json
    {
        "file_path": "/path/to/ckp.json",
        "variables": {
            "account_number": "123456789",
            "account_type": "checking",
            "state": "CA"
        },
        "config": {
            "mcp_server_url": "http://localhost:8000",
            "max_retries": 5
        }
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        # Load CKP file
        ckp_reader = CKPReader(request.file_path, request.variables)
        
        # Validate variables
        validation = ckp_reader.validate_variables(request.variables)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Variable validation failed: {', '.join(validation['errors'])}"
            )
        
        # Get steps
        steps = ckp_reader.get_steps()
        
        # Execute
        agent = get_agent(request.config)
        result = await agent.execute(
            steps,
            ckp_reader=ckp_reader,
            variables=request.variables
        )
        
        execution_time = time.time() - start_time
        
        return ExecutionResponse(
            success=result.get("success", False),
            total_steps=result.get("total_steps", 0),
            successful_steps=result.get("successful_steps", 0),
            failed_steps=result.get("failed_steps", 0),
            results=result.get("results", []),
            execution_time=execution_time
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


class PlanRequest(BaseModel):
    """Request model for execution plan."""
    file_path: str = Field(..., description="Path to JSON/CKP file")
    variables: Optional[Dict[str, Any]] = Field(None, description="Optional variables for CKP files")


@app.post("/api/v1/plan", response_model=ExecutionPlanResponse, tags=["Planning"])
async def get_execution_plan(request: PlanRequest):
    """
    Get execution plan for a file without executing.
    
    Request body:
    ```json
    {
        "file_path": "/path/to/ckp.json",
        "variables": {
            "account_number": "123456789"
        }
    }
    ```
    
    Returns execution plan with steps, rules, validations, etc.
    """
    try:
        import json
        from pathlib import Path
        
        path = Path(request.file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Check if it's CKP format
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "procedure_id" in data and "base_steps" in data:
            # CKP format
            ckp_reader = CKPReader(request.file_path, request.variables or {})
            steps = ckp_reader.get_steps()
            
            return ExecutionPlanResponse(
                procedure_id=ckp_reader.get_procedure_id(),
                description=ckp_reader.get_description(),
                total_steps=len(steps),
                variables=request.variables or {},
                steps=steps,
                rules=ckp_reader.get_rules(),
                validations=ckp_reader.get_validations(),
                exceptions=ckp_reader.get_exceptions(),
                recovery_flows=ckp_reader.get_recovery_flows()
            )
        else:
            # Simple JSON format
            steps = StepReader.load_steps(request.file_path)
            
            return ExecutionPlanResponse(
                total_steps=len(steps),
                variables=request.variables or {},
                steps=steps
            )
    except Exception as e:
        logger.error(f"Failed to get execution plan: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get execution plan: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
