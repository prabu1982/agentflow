"""
Agent that reads JSON configuration and executes MCP tools using LangGraph and FAST MCP.
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Optional, TypedDict

from fastmcp import Client
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    steps: List[Dict[str, Any]]
    current_step_index: int
    results: List[Dict[str, Any]]


class StepExecutorAgent:
    """Agent that executes steps from JSON configuration using MCP tools."""
    
    def __init__(self, mcp_server_url: Optional[str] = None, mcp_server_name: str = "mcp-server"):
        """
        Initialize the agent with MCP client.
        
        Args:
            mcp_server_url: URL of the MCP server (optional, can be configured via env)
            mcp_server_name: Name identifier for the MCP server
        """
        self.mcp_server_name = mcp_server_name
        # Normalize server URL - remove trailing /mcp/ if present, then add /mcp
        if mcp_server_url:
            mcp_server_url = mcp_server_url.rstrip('/mcp/').rstrip('/mcp').rstrip('/')
            if not mcp_server_url.endswith('/mcp'):
                mcp_server_url = f"{mcp_server_url}/mcp"
        self.mcp_server_url = mcp_server_url or "http://127.0.0.1:8000/mcp"
        self.graph = self._build_graph()
    
    def _resolve_variables(self, value: str, inputs: Dict[str, Any]) -> str:
        """
        Resolve variables in the format {{variable_name}} from inputs.
        
        Args:
            value: String that may contain variables like {{url}}
            inputs: Dictionary of input variables
            
        Returns:
            Resolved string with variables replaced
        """
        if not isinstance(value, str):
            return value
        
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, value)
        
        resolved = value
        for var_name in matches:
            if var_name in inputs:
                resolved = resolved.replace(f'{{{{{var_name}}}}}', str(inputs[var_name]))
            else:
                raise ValueError(f"Variable {var_name} not found in inputs")
        
        return resolved
    
    def _prepare_tool_arguments(self, step: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare arguments for MCP tool call by resolving variables.
        
        Args:
            step: Step configuration dictionary
            inputs: Input variables dictionary
            
        Returns:
            Dictionary of resolved arguments for the MCP tool
        """
        args = {}
        
        # Resolve 'value' attribute
        if 'value' in step:
            args['value'] = self._resolve_variables(step['value'], inputs)
        
        # Resolve 'xpath' attribute
        if 'xpath' in step:
            args['xpath'] = self._resolve_variables(step['xpath'], inputs)
        
        # Include any other attributes as arguments
        for key, val in step.items():
            if key not in ['action'] and key not in args:
                if isinstance(val, str):
                    args[key] = self._resolve_variables(val, inputs)
                else:
                    args[key] = val
        
        return args
    
    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool using FastMCP's Client class.
        This uses the same approach as mcp_client.py.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
        
        Returns:
            Result from the tool call
        """
        async def _async_call():
            """Async wrapper to call the tool."""
            client = Client(self.mcp_server_url)
            async with client:
                result = await client.call_tool(tool_name, arguments)
                # Extract data from result (same as mcp_client.py)
                if hasattr(result, 'data'):
                    return result.data
                elif hasattr(result, 'content'):
                    return result.content
                elif isinstance(result, dict):
                    return result.get('data', result.get('result', result))
                else:
                    return result
        
        # Run the async call synchronously
        return asyncio.run(_async_call())
    
    def _execute_step(self, state: AgentState) -> AgentState:
        """
        Execute a single step by calling the MCP tool.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        if state["current_step_index"] >= len(state["steps"]):
            return state
        
        step = state["steps"][state["current_step_index"]]
        action = step.get('action')
        
        if not action:
            raise ValueError(f"Step {state['current_step_index']} missing 'action' attribute")
        
        # Prepare arguments
        tool_args = self._prepare_tool_arguments(step, state["inputs"])
        
        # Execute MCP tool using FastMCP's Client class and call_tool method
        try:
            # Use FastMCP Client class to call tools (same as mcp_client.py)
            result = self._call_mcp_tool(action, tool_args)
            
            state["results"].append({
                'step_index': state["current_step_index"],
                'action': action,
                'arguments': tool_args,
                'result': result,
                'status': 'success'
            })
        except Exception as e:
            state["results"].append({
                'step_index': state["current_step_index"],
                'action': action,
                'arguments': tool_args,
                'error': str(e),
                'status': 'error'
            })
            raise
        
        state["current_step_index"] += 1
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if we should continue executing steps or end.
        
        Args:
            state: Current agent state
            
        Returns:
            'continue' if more steps, 'end' otherwise
        """
        if state["current_step_index"] >= len(state["steps"]):
            return 'end'
        return 'continue'
    
    def _print_step_summary(self, config: Dict[str, Any], inputs: Dict[str, Any], steps: List[Dict[str, Any]]) -> None:
        """
        Print a summary of the steps that will be executed.
        
        Args:
            config: Configuration dictionary
            inputs: Resolved input variables
            steps: List of steps to execute
        """
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        
        # Agent information
        agent_name = config.get('agent', 'Unknown')
        description = config.get('description', 'No description')
        agent_type = config.get('type', 'Unknown')
        
        print(f"\nAgent: {agent_name}")
        print(f"Type: {agent_type}")
        print(f"Description: {description}")
        
        # Input variables
        print(f"\nInput Variables:")
        if inputs:
            for key, value in inputs.items():
                print(f"  {key}: {value}")
        else:
            print("  (none)")
        
        # Steps summary
        print(f"\nSteps to Execute ({len(steps)} total):")
        print("-" * 70)
        
        for idx, step in enumerate(steps, 1):
            action = step.get('action', 'Unknown')
            
            # Prepare resolved arguments for display
            try:
                resolved_args = self._prepare_tool_arguments(step, inputs)
            except Exception as e:
                resolved_args = {"error": f"Could not resolve: {str(e)}"}
            
            print(f"\nStep {idx}: {action}")
            
            # Show original step configuration
            if 'value' in step:
                original_value = step['value']
                resolved_value = resolved_args.get('value', original_value)
                if original_value != resolved_value:
                    print(f"  value: {original_value} → {resolved_value}")
                else:
                    print(f"  value: {resolved_value}")
            
            if 'xpath' in step:
                original_xpath = step['xpath']
                resolved_xpath = resolved_args.get('xpath', original_xpath)
                if original_xpath != resolved_xpath:
                    print(f"  xpath: {original_xpath} → {resolved_xpath}")
                else:
                    print(f"  xpath: {resolved_xpath}")
            
            # Show any other arguments
            for key, val in step.items():
                if key not in ['action', 'value', 'xpath']:
                    resolved_val = resolved_args.get(key, val)
                    if isinstance(val, str) and val != resolved_val:
                        print(f"  {key}: {val} → {resolved_val}")
                    else:
                        print(f"  {key}: {resolved_val}")
        
        print("\n" + "=" * 70)
        print("Starting execution...")
        print("=" * 70 + "\n")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for step execution.
        
        Returns:
            Compiled LangGraph graph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("execute_step", self._execute_step)
        
        # Set entry point
        workflow.set_entry_point("execute_step")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "execute_step",
            self._should_continue,
            {
                "continue": "execute_step",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Parsed configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def run(self, config: Dict[str, Any], additional_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent with the given configuration.
        
        Args:
            config: Configuration dictionary (can be loaded from JSON)
            additional_inputs: Additional input variables to merge with config inputs
            
        Returns:
            Final state with execution results
        """
        # Extract configuration components
        inputs = config.get('inputs', {}).copy()
        if additional_inputs:
            inputs.update(additional_inputs)
        
        steps = config.get('steps', [])
        
        if not steps:
            raise ValueError("Configuration must contain 'steps' array")
        
        # Print summary of steps before execution
        self._print_step_summary(config, inputs, steps)
        
        # Initialize state
        initial_state: AgentState = {
            "config": config,
            "inputs": inputs,
            "steps": steps,
            "current_step_index": 0,
            "results": []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            'agent': config.get('agent', 'Unknown'),
            'description': config.get('description', ''),
            'next_node': config.get('next_node', 'end'),
            'results': final_state["results"],
            'total_steps': len(steps),
            'completed_steps': final_state["current_step_index"]
        }


def create_agent(mcp_server_url: Optional[str] = None, mcp_server_name: str = "mcp-server") -> StepExecutorAgent:
    """
    Factory function to create a StepExecutorAgent instance.
    
    Args:
        mcp_server_url: URL of the MCP server
        mcp_server_name: Name identifier for the MCP server
    
    Returns:
        StepExecutorAgent instance
    """
    return StepExecutorAgent(mcp_server_url=mcp_server_url, mcp_server_name=mcp_server_name)