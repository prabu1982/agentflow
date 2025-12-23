"""Mainframe Agent using LangGraph, MCP Server, and OLLAMA for error recovery with CKP support."""

import logging
import asyncio
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END

from config import AgentConfig
from mcp_client import MCPClient
from ollama_client import OllamaClient
from step_reader import StepReader
from ckp_reader import CKPReader

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Mainframe Agent."""
    steps: List[Dict[str, Any]]
    current_step_index: int
    execution_results: List[Dict[str, Any]]
    context: Dict[str, Any]
    retry_count: int
    error_fix: Optional[Dict[str, Any]]
    decision: Optional[str]
    ckp_reader: Optional[CKPReader]
    variables: Dict[str, Any]
    detected_exception: Optional[str]
    recovery_flow_executed: bool


class MainframeAgent:
    """Mainframe Agent with LangGraph orchestration, MCP execution, and OLLAMA error recovery."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize Mainframe Agent.
        
        Args:
            config: Agent configuration (uses default if not provided)
        """
        self.config = config or AgentConfig()
        
        # Initialize clients
        self.mcp_client = MCPClient(self.config)
        self.ollama_client = OllamaClient(self.config)
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("Mainframe Agent initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for step execution with error recovery."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("prepare_step", self._prepare_step)
        workflow.add_node("apply_rules", self._apply_rules)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("validate_step", self._validate_step)
        workflow.add_node("check_exceptions", self._check_exceptions)
        workflow.add_node("execute_recovery", self._execute_recovery)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("apply_fix", self._apply_fix)
        workflow.add_node("make_decision", self._make_decision)
        
        # Set entry point
        workflow.set_entry_point("prepare_step")
        
        # Add edges
        workflow.add_edge("prepare_step", "apply_rules")
        workflow.add_edge("apply_rules", "execute_step")
        
        workflow.add_conditional_edges(
            "execute_step",
            self._check_execution_result,
            {
                "success": "validate_step",
                "error": "check_exceptions",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_step",
            self._check_validation_result,
            {
                "pass": "make_decision",
                "fail": "check_exceptions",
                "no_validation": "make_decision"
            }
        )
        
        workflow.add_conditional_edges(
            "check_exceptions",
            self._check_exception_result,
            {
                "exception": "execute_recovery",
                "no_exception": "handle_error",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "execute_recovery",
            self._check_recovery_result,
            {
                "success": "make_decision",
                "retry": "apply_rules",
                "fail": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self._check_retry_eligibility,
            {
                "retry": "apply_fix",
                "skip": "make_decision",
                "stop": END
            }
        )
        
        workflow.add_edge("apply_fix", "apply_rules")
        
        workflow.add_conditional_edges(
            "make_decision",
            self._decision_route,
            {
                "next": "prepare_step",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _prepare_step(self, state: AgentState) -> AgentState:
        """Prepare the current step for execution."""
        steps = state["steps"]
        current_index = state["current_step_index"]
        
        if current_index >= len(steps):
            state["decision"] = "end"
            return state
        
        current_step = steps[current_index]
        step_id = current_step.get("step_id", f"S{current_index + 1}")
        logger.info(f"Preparing step {step_id} ({current_index + 1}/{len(steps)}): {current_step.get('action')}")
        
        # Reset retry count if starting a new step
        if state.get("retry_count", 0) == 0:
            state["retry_count"] = 0
            state["recovery_flow_executed"] = False
            state["detected_exception"] = None
        
        # Add step to context
        state["context"]["current_step"] = current_step
        state["context"]["step_index"] = current_index
        state["context"]["step_id"] = step_id
        
        return state
    
    async def _apply_rules(self, state: AgentState) -> AgentState:
        """Apply business rules to the current step."""
        ckp_reader = state.get("ckp_reader")
        if not ckp_reader:
            return state
        
        current_step = state["context"]["current_step"]
        step_id = current_step.get("step_id")
        variables = state.get("variables", {})
        
        # Get rule for this step
        rule = ckp_reader.get_rule_for_step(step_id)
        if not rule:
            return state
        
        logger.info(f"Applying rule {rule.get('rule_id')} to step {step_id}")
        
        # Check condition
        condition = rule.get("condition", {})
        if self._evaluate_condition(condition, variables):
            # Apply constraint or effect
            if "constraint" in rule:
                constraint = rule["constraint"]
                field = constraint.get("field")
                allowed_values = constraint.get("allowed_values", [])
                
                if field in variables and variables[field] not in allowed_values:
                    severity = rule.get("severity", "soft_prompt")
                    if severity == "hard_block":
                        logger.error(
                            f"Rule {rule.get('rule_id')} violation: {field} must be one of {allowed_values}"
                        )
                        state["context"]["rule_violation"] = {
                            "rule_id": rule.get("rule_id"),
                            "field": field,
                            "message": f"{field} must be one of {allowed_values}"
                        }
                        # For hard blocks, we'll let OLLAMA fix it
                    else:
                        logger.warning(f"Rule {rule.get('rule_id')} soft violation for {field}")
            
            if "effect" in rule:
                effect = rule["effect"]
                if "set_default" in effect:
                    for field, value in effect["set_default"].items():
                        if field not in variables:
                            variables[field] = value
                            logger.info(f"Rule {rule.get('rule_id')} set default: {field} = {value}")
        
        return state
    
    def _evaluate_condition(self, condition: Dict[str, Any], variables: Dict[str, Any]) -> bool:
        """Evaluate a condition against variables."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if field not in variables:
            if operator == "missing":
                return value is True
            return False
        
        field_value = variables[field]
        
        if operator == "equals":
            return field_value == value
        elif operator == "in":
            return field_value in value if isinstance(value, list) else False
        elif operator == "not_equals":
            return field_value != value
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    async def _execute_step(self, state: AgentState) -> AgentState:
        """Execute the current step using MCP client."""
        steps = state["steps"]
        current_index = state["current_step_index"]
        
        if current_index >= len(steps):
            state["decision"] = "end"
            return state
        
        current_step = steps[current_index]
        
        # If a fix was applied, use the fixed step
        if state.get("error_fix") and state["error_fix"].get("success"):
            current_step = state["error_fix"]["updated_step"]
            logger.info(f"Using fixed step for retry: {current_step.get('action')}")
        
        logger.info(f"Executing step {current_index + 1}/{len(steps)}: {current_step.get('action')}")
        
        # Convert CKP step format to MCP tool format
        mcp_step = self._convert_ckp_to_mcp(current_step)
        
        # Execute step via MCP
        result = await self.mcp_client.execute_step(mcp_step)
        
        # Store result
        result["step_index"] = current_index
        result["step"] = current_step
        result["step_id"] = current_step.get("step_id")
        state["execution_results"].append(result)
        state["context"]["last_result"] = result
        state["context"]["last_mcp_result"] = result.get("result", {})
        
        # Update step index on success
        if result.get("success"):
            state["current_step_index"] = current_index + 1
            state["retry_count"] = 0
            state["error_fix"] = None
        else:
            # Increment retry count
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def _convert_ckp_to_mcp(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CKP step format to MCP tool format."""
        action = step.get("action")
        
        # Map CKP actions to MCP tools
        tool_mapping = {
            "launch_application": "launch_app",
            "wait_for_element": "wait_for_element",
            "send_keys": "send_keys",
            "click": "click",
            "type": "type"
        }
        
        tool = tool_mapping.get(action, action)
        
        # Build MCP arguments
        arguments = {
            "action": action,
            "element": step.get("element"),
            "value": step.get("value"),
            "application_type": step.get("application_type"),
            "special_keys": step.get("special_keys"),
            "windowName": step.get("windowName"),
            "fieldName": step.get("fieldName"),
            "screenName": step.get("screenName"),
            "startX": step.get("startX"),
            "startY": step.get("startY"),
            "endX": step.get("endX"),
            "endY": step.get("endY")
        }
        
        # Remove empty values
        arguments = {k: v for k, v in arguments.items() if v not in [None, "", []]}
        
        return {
            "tool": tool,
            "arguments": arguments
        }
    
    async def _validate_step(self, state: AgentState) -> AgentState:
        """Validate step execution result."""
        ckp_reader = state.get("ckp_reader")
        if not ckp_reader:
            state["context"]["validation_result"] = "no_validation"
            return state
        
        current_step = state["context"]["current_step"]
        step_id = current_step.get("step_id")
        last_result = state["context"].get("last_result", {})
        
        # Get validation for this step
        validation = ckp_reader.get_validation_for_step(step_id)
        if not validation:
            state["context"]["validation_result"] = "no_validation"
            return state
        
        logger.info(f"Validating step {step_id} with validation {validation.get('validation_id')}")
        
        # Perform validation
        validation_type = validation.get("type")
        mcp_result = state["context"].get("last_mcp_result", {})
        
        if validation_type == "screen_text_check":
            expected_text_any = validation.get("expected_text_any", [])
            screen_text = str(mcp_result.get("screen_text", mcp_result.get("text", "")))
            
            found = False
            for expected in expected_text_any:
                if expected.lower() in screen_text.lower():
                    found = True
                    break
            
            if found:
                logger.info(f"Validation {validation.get('validation_id')} passed")
                state["context"]["validation_result"] = "pass"
            else:
                logger.warning(
                    f"Validation {validation.get('validation_id')} failed: "
                    f"Expected one of {expected_text_any}, got: {screen_text[:100]}"
                )
                state["context"]["validation_result"] = "fail"
                state["context"]["validation_error"] = {
                    "validation_id": validation.get("validation_id"),
                    "expected": expected_text_any,
                    "actual": screen_text[:200]
                }
                # Check if this triggers an exception
                exception_code = validation.get("failure_routes_to_exception")
                if exception_code:
                    state["detected_exception"] = exception_code
        else:
            logger.warning(f"Unknown validation type: {validation_type}")
            state["context"]["validation_result"] = "no_validation"
        
        return state
    
    async def _check_exceptions(self, state: AgentState) -> AgentState:
        """Check for exceptions in execution result."""
        ckp_reader = state.get("ckp_reader")
        if not ckp_reader:
            return state
        
        # Check if exception was already detected in validation
        if state.get("detected_exception"):
            exception_code = state["detected_exception"]
            logger.warning(f"Exception detected: {exception_code}")
            return state
        
        # Check execution result for exception triggers
        last_result = state["context"].get("last_result", {})
        mcp_result = state["context"].get("last_mcp_result", {})
        screen_text = str(mcp_result.get("screen_text", mcp_result.get("text", "")))
        
        exceptions = ckp_reader.get_exceptions()
        for exception in exceptions:
            trigger = exception.get("trigger", {})
            trigger_type = trigger.get("type")
            
            if trigger_type == "screen_text":
                contains = trigger.get("contains")
                contains_any = trigger.get("contains_any", [])
                
                if contains and contains.lower() in screen_text.lower():
                    exception_code = exception.get("exception_code")
                    logger.warning(f"Exception detected: {exception_code}")
                    state["detected_exception"] = exception_code
                    return state
                
                if contains_any:
                    for text in contains_any:
                        if text.lower() in screen_text.lower():
                            exception_code = exception.get("exception_code")
                            logger.warning(f"Exception detected: {exception_code}")
                            state["detected_exception"] = exception_code
                            return state
        
        return state
    
    async def _execute_recovery(self, state: AgentState) -> AgentState:
        """Execute recovery flow for detected exception."""
        ckp_reader = state.get("ckp_reader")
        exception_code = state.get("detected_exception")
        
        if not ckp_reader or not exception_code:
            return state
        
        # Check if recovery already executed
        if state.get("recovery_flow_executed"):
            logger.warning(f"Recovery flow already executed for {exception_code}")
            state["context"]["recovery_result"] = "fail"
            return state
        
        # Get recovery flow
        recovery = ckp_reader.get_recovery_for_exception(exception_code)
        if not recovery:
            logger.error(f"No recovery flow found for exception: {exception_code}")
            state["context"]["recovery_result"] = "fail"
            return state
        
        logger.info(f"Executing recovery flow {recovery.get('recovery_id')} for {exception_code}")
        
        # Check required secrets
        required_secrets = recovery.get("required_secrets", [])
        variables = state.get("variables", {})
        missing_secrets = [s for s in required_secrets if s not in variables]
        if missing_secrets:
            logger.error(f"Missing required secrets for recovery: {missing_secrets}")
            state["context"]["recovery_result"] = "fail"
            return state
        
        # Execute recovery steps
        recovery_steps = recovery.get("steps", [])
        variables = state.get("variables", {})
        
        # Substitute variables in recovery steps
        for recovery_step in recovery_steps:
            if "value" in recovery_step and isinstance(recovery_step["value"], str):
                pattern = r'\{\{(\w+)\}\}'
                matches = re.findall(pattern, recovery_step["value"])
                for var_name in matches:
                    if var_name in variables:
                        recovery_step["value"] = recovery_step["value"].replace(
                            f"{{{{{var_name}}}}}", str(variables[var_name])
                        )
        
        # Execute recovery steps via MCP
        recovery_results = []
        for recovery_step in recovery_steps:
            mcp_step = self._convert_ckp_to_mcp(recovery_step)
            result = await self.mcp_client.execute_step(mcp_step)
            recovery_results.append(result)
            
            if not result.get("success"):
                logger.error(f"Recovery step failed: {result.get('message')}")
                state["context"]["recovery_result"] = "fail"
                return state
        
        logger.info(f"Recovery flow {recovery.get('recovery_id')} completed successfully")
        state["recovery_flow_executed"] = True
        state["context"]["recovery_result"] = "success"
        state["context"]["recovery_results"] = recovery_results
        
        # Reset exception after recovery
        state["detected_exception"] = None
        
        return state
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors by requesting fix from OLLAMA."""
        current_index = state["current_step_index"]
        steps = state["steps"]
        current_step = steps[current_index]
        last_result = state["context"].get("last_result", {})
        retry_count = state.get("retry_count", 0)
        
        logger.warning(
            f"Step {current_index + 1} failed (retry {retry_count}/{self.config.max_retries}): "
            f"{last_result.get('message', 'Unknown error')}"
        )
        
        # Check if we should retry
        if retry_count > self.config.max_retries:
            logger.error(f"Max retries ({self.config.max_retries}) exceeded for step {current_index + 1}")
            state["decision"] = "stop"
            return state
        
        # Request fix from OLLAMA
        logger.info("Requesting error fix from OLLAMA...")
        fix_result = await self.ollama_client.generate_fix(
            step=current_step,
            error=last_result,
            context=state["context"]
        )
        
        if fix_result.get("success"):
            state["error_fix"] = fix_result
            logger.info("Received fix suggestion from OLLAMA")
        else:
            logger.error(f"Failed to get fix from OLLAMA: {fix_result.get('message')}")
            state["error_fix"] = {
                "success": False,
                "updated_step": current_step
            }
        
        return state
    
    async def _apply_fix(self, state: AgentState) -> AgentState:
        """Apply the fix suggested by OLLAMA."""
        if state.get("error_fix") and state["error_fix"].get("success"):
            updated_step = state["error_fix"]["updated_step"]
            current_index = state["current_step_index"]
            
            # Update the step in the steps list
            state["steps"][current_index] = updated_step
            
            logger.info(
                f"Applied fix to step {current_index + 1}: "
                f"{state['error_fix'].get('fix_suggestion', 'Fix applied')[:100]}"
            )
        
        # Add delay before retry
        await asyncio.sleep(self.config.retry_delay)
        
        return state
    
    async def _make_decision(self, state: AgentState) -> AgentState:
        """Make decision about next action."""
        current_index = state["current_step_index"]
        steps = state["steps"]
        
        # Check if all steps are complete
        if current_index >= len(steps):
            state["decision"] = "end"
            logger.info("All steps completed successfully")
            return state
        
        # Continue to next step
        state["decision"] = "next"
        return state
    
    def _check_execution_result(self, state: AgentState) -> str:
        """Check execution result and route accordingly."""
        current_index = state["current_step_index"]
        steps = state["steps"]
        last_result = state["context"].get("last_result", {})
        
        # Check if all steps are complete
        if current_index >= len(steps):
            return "end"
        
        # Check if last execution was successful
        if last_result.get("success"):
            return "success"
        
        # Execution failed
        return "error"
    
    def _check_validation_result(self, state: AgentState) -> str:
        """Check validation result and route accordingly."""
        validation_result = state["context"].get("validation_result", "no_validation")
        
        if validation_result == "pass":
            return "pass"
        elif validation_result == "fail":
            return "fail"
        else:
            return "no_validation"
    
    def _check_exception_result(self, state: AgentState) -> str:
        """Check exception result and route accordingly."""
        current_index = state["current_step_index"]
        steps = state["steps"]
        
        if current_index >= len(steps):
            return "end"
        
        if state.get("detected_exception"):
            return "exception"
        
        return "no_exception"
    
    def _check_recovery_result(self, state: AgentState) -> str:
        """Check recovery result and route accordingly."""
        recovery_result = state["context"].get("recovery_result", "fail")
        
        if recovery_result == "success":
            # Recovery successful, continue to next step
            return "success"
        elif recovery_result == "retry":
            # Retry the step
            return "retry"
        else:
            # Recovery failed, try OLLAMA fix
            return "fail"
    
    def _check_retry_eligibility(self, state: AgentState) -> str:
        """Check if step should be retried."""
        retry_count = state.get("retry_count", 0)
        current_step = state["steps"][state["current_step_index"]]
        
        # Check max retries
        if retry_count > self.config.max_retries:
            # Check step's on_error behavior (if available)
            on_error = current_step.get("on_error", "stop")
            if on_error == "skip":
                return "skip"
            return "stop"
        
        # Retry with fix
        return "retry"
    
    def _decision_route(self, state: AgentState) -> str:
        """Route based on decision."""
        decision = state.get("decision", "next")
        
        if decision == "end":
            return "end"
        
        return "next"
    
    async def execute(self, steps: List[Dict[str, Any]], ckp_reader: Optional[CKPReader] = None, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute automation steps using LangGraph workflow.
        
        Args:
            steps: List of automation steps
            ckp_reader: Optional CKP reader for advanced features
            variables: Optional variables for substitution
            
        Returns:
            Execution results and summary
        """
        # Initialize state
        initial_state: AgentState = {
            "steps": steps,
            "current_step_index": 0,
            "execution_results": [],
            "context": {},
            "retry_count": 0,
            "error_fix": None,
            "decision": None,
            "ckp_reader": ckp_reader,
            "variables": variables or {},
            "detected_exception": None,
            "recovery_flow_executed": False
        }
        
        logger.info(f"Starting execution of {len(steps)} steps")
        
        # Run workflow
        try:
            final_state = await self._run_workflow_async(initial_state)
            
            # Compile results
            results = final_state["execution_results"]
            successful = sum(1 for r in results if r.get("success", False))
            failed = len(results) - successful
            
            summary = {
                "success": failed == 0,
                "total_steps": len(steps),
                "successful_steps": successful,
                "failed_steps": failed,
                "results": results,
                "final_state": {
                    "current_step_index": final_state["current_step_index"],
                    "decision": final_state.get("decision")
                }
            }
            
            logger.info(
                f"Execution completed: {successful}/{len(steps)} steps successful"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "results": initial_state["execution_results"]
            }
    
    async def _run_workflow_async(self, initial_state: AgentState) -> AgentState:
        """Run the workflow with async support."""
        state = initial_state.copy()
        
        while True:
            # Prepare step
            state = await self._prepare_step(state)
            
            if state.get("decision") == "end":
                break
            
            # Apply rules
            state = await self._apply_rules(state)
            
            # Execute step
            state = await self._execute_step(state)
            
            # Check result
            route = self._check_execution_result(state)
            
            if route == "end":
                break
            elif route == "success":
                # Validate step
                state = await self._validate_step(state)
                validation_route = self._check_validation_result(state)
                
                if validation_route == "pass" or validation_route == "no_validation":
                    # Make decision
                    state = await self._make_decision(state)
                    if state.get("decision") == "end":
                        break
                    continue
                else:
                    # Validation failed, check for exceptions
                    route = "error"
            
            if route == "error":
                # Check for exceptions
                state = await self._check_exceptions(state)
                exception_route = self._check_exception_result(state)
                
                if exception_route == "exception":
                    # Execute recovery
                    state = await self._execute_recovery(state)
                    recovery_route = self._check_recovery_result(state)
                    
                    if recovery_route == "success":
                        # Recovery successful, continue
                        state = await self._make_decision(state)
                        if state.get("decision") == "end":
                            break
                        continue
                    elif recovery_route == "retry":
                        # Retry the step
                        continue
                    else:
                        # Recovery failed, try OLLAMA
                        route = "error"
                
                if route == "error":
                    # Handle error with OLLAMA
                    state = await self._handle_error(state)
                    
                    retry_route = self._check_retry_eligibility(state)
                    
                    if retry_route == "retry":
                        # Apply fix and retry
                        state = await self._apply_fix(state)
                        continue
                    elif retry_route == "skip":
                        # Skip to next step
                        state["current_step_index"] = state["current_step_index"] + 1
                        state["retry_count"] = 0
                        state = await self._make_decision(state)
                        if state.get("decision") == "end":
                            break
                        continue
                    else:
                        # Stop execution
                        break
        
        return state
    
    async def execute_from_file(self, file_path: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute steps from a JSON file (supports both CKP and simple JSON formats).
        
        Args:
            file_path: Path to JSON file containing steps
            variables: Optional variables for CKP files
            
        Returns:
            Execution results and summary
        """
        path = Path(file_path)
        
        # Check if it's a CKP file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "procedure_id" in data and "base_steps" in data:
                # It's a CKP file
                logger.info("Detected CKP format file")
                ckp_reader = CKPReader(file_path, variables)
                
                # Validate variables
                validation = ckp_reader.validate_variables(variables or {})
                if not validation["valid"]:
                    logger.error(f"Variable validation failed: {validation['errors']}")
                    raise ValueError(f"Variable validation failed: {', '.join(validation['errors'])}")
                
                steps = ckp_reader.get_steps()
                return await self.execute(steps, ckp_reader=ckp_reader, variables=variables or {})
            else:
                # Simple JSON format
                logger.info("Detected simple JSON format file")
                steps = StepReader.load_steps(file_path)
                return await self.execute(steps)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Close all clients."""
        await self.mcp_client.close()
        await self.ollama_client.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
