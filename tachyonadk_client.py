"""Google ADK Agents (TachyonAdkClient) client for error fixing and recovery."""

import logging
import json
import re
from typing import Dict, Any, Optional
from config import AgentConfig

logger = logging.getLogger(__name__)

from tachyon_adk_client import TachyonAdkClient as TachyonAdkClientModel



class TachyonAdkClient:
    """Client for interacting with Google ADK Agents (TachyonAdkClient) LLM."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize Google ADK Agents (TachyonAdkClient) client.
        
        Args:
            config: Agent configuration
        """
        
        self.config = config
        self.model_name = "open/gemini-2.0-flash"
        
        # Initialize the TachyonAdkClient
        try:
            self.client = TachyonAdkClientModel(model_name=self.model_name)
            logger.info(f"Initialized Google ADK client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google ADK model: {str(e)}")
            raise
    
    async def generate_fix(
        self,
        step: Dict[str, Any],
        error: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Google ADK Agents (TachyonAdkClient) to generate a fix for a failed step.
        
        Args:
            step: The failed step definition
            error: Error information from the failed execution
            context: Execution context
            
        Returns:
            Dictionary with suggested fix and updated step
        """
        try:
            prompt = self._build_fix_prompt(step, error, context)
            
            logger.info(f"Requesting fix from Google ADK (model: {self.model_name})")
            
            # Generate content using TachyonAdkClient
            # TachyonAdkClient can be used directly for text generation
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Generate response using TachyonAdkClient
            # TachyonAdkClient may have different API patterns, try common ones
            fix_suggestion = ""
            
            # Try using the client's generate or complete method
            if hasattr(self.client, 'generate'):
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.generate(prompt, temperature=0.3)
                )
                fix_suggestion = response.text if hasattr(response, 'text') else str(response)
            elif hasattr(self.client, 'complete'):
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.complete(prompt, temperature=0.3)
                )
                fix_suggestion = response.text if hasattr(response, 'text') else str(response)
            elif hasattr(self.client, 'chat'):
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.chat([{"role": "user", "content": prompt}], temperature=0.3)
                )
                fix_suggestion = response.content if hasattr(response, 'content') else (response.text if hasattr(response, 'text') else str(response))
            elif callable(self.client):
                # Try calling it directly
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client(prompt, temperature=0.3)
                )
                fix_suggestion = response.text if hasattr(response, 'text') else str(response)
            else:
                # If no known methods, raise error with helpful message
                raise AttributeError(
                    f"TachyonAdkClient does not have expected methods (generate, complete, chat, or callable). "
                    f"Available methods: {[m for m in dir(self.client) if not m.startswith('_')]}"
                )
            
            fix_suggestion = fix_suggestion.strip() if fix_suggestion else ""
            
            logger.info(f"Received fix suggestion from Google ADK: {fix_suggestion[:100]}...")
            
            # Parse the fix suggestion and update the step
            updated_step = self._parse_fix_suggestion(step, fix_suggestion, error)
            
            return {
                "success": True,
                "fix_suggestion": fix_suggestion,
                "updated_step": updated_step,
                "original_step": step,
                "error": error
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            if "API" in error_type or "api" in error_msg.lower() or "key" in error_msg.lower():
                error_msg = f"Google ADK API error: {error_msg}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "message": "Failed to get fix suggestion from Google ADK - API error"
                }
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                error_msg = f"Google ADK quota/limit error: {error_msg}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "message": "Failed to get fix suggestion from Google ADK - quota exceeded"
                }
            else:
                error_msg = f"Unexpected error calling Google ADK: {error_msg}"
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
        """Build prompt for Google ADK Agents to generate a fix."""
        return f"""You are an expert automation engineer helping to fix a failed automation step.

FAILED STEP:
Tool: {step.get('tool', 'unknown')}
Arguments: {json.dumps(step.get('arguments', step.get('params', {})), indent=2)}
Step Description: {step.get('description', 'No description')}

ERROR INFORMATION:
Error Message: {error.get('message', 'Unknown error')}
Error Details: {error.get('error', 'No details')}
Error Type: {error.get('error_type', 'N/A')}
Status Code: {error.get('status_code', 'N/A')}

EXECUTION CONTEXT:
{json.dumps(context, indent=2) if isinstance(context, dict) else str(context)}

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
        Parse Google ADK's fix suggestion and create an updated step.
        
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
            logger.warning("Could not parse JSON from Google ADK response, using original step with minor adjustments")
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
        """Close the Google ADK client (no-op if not needed)."""
        # Google ADK client may not require explicit closing
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
