"""OLLAMA LLM client for error fixing and recovery."""

import logging
import httpx
from typing import Dict, Any, Optional
from config import AgentConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with OLLAMA LLM."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize OLLAMA client.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.base_url = config.ollama_url.rstrip('/')
        self.model = config.ollama_model
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0
        )
    
    async def generate_fix(
        self,
        step: Dict[str, Any],
        error: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use OLLAMA LLM to generate a fix for a failed step.
        
        Args:
            step: The failed step definition
            error: Error information from the failed execution
            context: Execution context
            
        Returns:
            Dictionary with suggested fix and updated step
        """
        try:
            prompt = self._build_fix_prompt(step, error, context)
            
            logger.info(f"Requesting fix from OLLAMA (model: {self.model})")
            
            response = await self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            fix_suggestion = result.get("response", "").strip()
            
            logger.info(f"Received fix suggestion from OLLAMA: {fix_suggestion[:100]}...")
            
            # Parse the fix suggestion and update the step
            updated_step = self._parse_fix_suggestion(step, fix_suggestion, error)
            
            return {
                "success": True,
                "fix_suggestion": fix_suggestion,
                "updated_step": updated_step,
                "original_step": step,
                "error": error
            }
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error calling OLLAMA: {e.response.text}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Failed to get fix suggestion from OLLAMA"
            }
            
        except httpx.RequestError as e:
            error_msg = f"Request error calling OLLAMA: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": f"Failed to connect to OLLAMA server: {error_msg}"
            }
            
        except Exception as e:
            error_msg = f"Unexpected error calling OLLAMA: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "message": f"Unexpected error: {error_msg}"
            }
    
    def _build_fix_prompt(
        self,
        step: Dict[str, Any],
        error: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for OLLAMA to generate a fix."""
        return f"""You are an expert automation engineer helping to fix a failed automation step.

FAILED STEP:
Tool: {step.get('tool', 'unknown')}
Arguments: {step.get('arguments', step.get('params', {}))}
Step Description: {step.get('description', 'No description')}

ERROR INFORMATION:
Error Message: {error.get('message', 'Unknown error')}
Error Details: {error.get('error', 'No details')}
Status Code: {error.get('status_code', 'N/A')}

EXECUTION CONTEXT:
{context}

TASK:
Analyze the error and provide a fix. Your response should:
1. Explain what went wrong
2. Suggest specific changes to the step arguments/parameters
3. Provide the corrected step in JSON format

IMPORTANT: Respond with a JSON object containing:
{{
    "analysis": "Brief explanation of what went wrong",
    "fix_reasoning": "Why this fix should work",
    "updated_arguments": {{
        // Updated arguments/parameters for the step
    }}
}}

Only return the JSON object, no additional text."""

    def _parse_fix_suggestion(
        self,
        original_step: Dict[str, Any],
        fix_suggestion: str,
        error: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse OLLAMA's fix suggestion and create an updated step.
        
        Args:
            original_step: Original step definition
            fix_suggestion: LLM's fix suggestion
            error: Error information
            
        Returns:
            Updated step with fixed arguments
        """
        import json
        import re
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^{}]*"updated_arguments"[^{}]*\{[^{}]*\}[^{}]*\}', fix_suggestion, re.DOTALL)
            if json_match:
                fix_json = json.loads(json_match.group())
                updated_args = fix_json.get("updated_arguments", {})
            else:
                # Try to parse the entire response as JSON
                fix_json = json.loads(fix_suggestion)
                updated_args = fix_json.get("updated_arguments", {})
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, try to extract key-value pairs from text
            logger.warning("Could not parse JSON from OLLAMA response, using original step with minor adjustments")
            updated_args = original_step.get("arguments", original_step.get("params", {}))
        
        # Create updated step
        updated_step = original_step.copy()
        if "arguments" in updated_step:
            updated_step["arguments"].update(updated_args)
        elif "params" in updated_step:
            updated_step["params"].update(updated_args)
        else:
            updated_step["arguments"] = updated_args
        
        # Add metadata about the fix
        updated_step["_fix_applied"] = True
        updated_step["_fix_reasoning"] = fix_json.get("fix_reasoning", "Fix applied based on error analysis") if 'fix_json' in locals() else "Fix applied"
        updated_step["_original_error"] = error.get("message", "Unknown error")
        
        return updated_step
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
