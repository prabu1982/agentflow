"""
Main entry point for the Step Executor Agent.
"""

import json
import sys
from pathlib import Path
from web_agent import StepExecutorAgent, create_agent


def main():
    """Main function to run the agent."""
    # Default config file
    config_file = "web_steps.json"
    
    # Allow config file to be passed as command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # MCP server configuration (can be set via environment variables)
    mcp_server_url = "http://localhost:8000/mcp" # Set this to your MCP server URL
    mcp_server_name = "mcp-server"  # Name of the MCP server
    
    # Additional inputs that can override or extend config inputs
    additional_inputs = {}
    
    # Check for user_id in environment or command line
    if len(sys.argv) > 2:
        additional_inputs['user_id'] = sys.argv[2]
    
    try:
        # Create agent
        agent = create_agent(
            mcp_server_url=mcp_server_url,
            mcp_server_name=mcp_server_name
        )
        
        # Load configuration
        config = agent.load_config(config_file)
        
        print(f"Agent: {config.get('agent', 'Unknown')}")
        print(f"Description: {config.get('description', '')}")
        print(f"Steps to execute: {len(config.get('steps', []))}")
        print("-" * 50)
        
        # Run the agent
        result = agent.run(config, additional_inputs=additional_inputs)
        
        # Print results
        print("\nExecution Results:")
        print(f"Total Steps: {result['total_steps']}")
        print(f"Completed Steps: {result['completed_steps']}")
        print(f"Next Node: {result['next_node']}")
        print("\nStep Details:")
        
        for step_result in result['results']:
            print(f"\n  Step {step_result['step_index'] + 1}:")
            print(f"    Action: {step_result['action']}")
            print(f"    Arguments: {step_result['arguments']}")
            print(f"    Status: {step_result['status']}")
            if step_result['status'] == 'success':
                print(f"    Result: {step_result.get('result', 'N/A')}")
            else:
                print(f"    Error: {step_result.get('error', 'Unknown error')}")
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())