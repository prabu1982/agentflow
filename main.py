"""Main entry point for Mainframe Agent."""

import asyncio
import logging
import sys
import argparse
import json
import time
from pathlib import Path

from config import AgentConfig
from mainframe_agent import MainframeAgent
from ckp_reader import CKPReader
from step_reader import StepReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_variables_from_file(file_path: str) -> dict:
    """Load variables from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading variables file {file_path}: {str(e)}")
        raise


def parse_variable_args(variable_args: list) -> dict:
    """Parse variable arguments in format key=value."""
    variables = {}
    for var_arg in variable_args:
        if '=' not in var_arg:
            logger.warning(f"Invalid variable format: {var_arg}. Expected key=value")
            continue
        key, value = var_arg.split('=', 1)
        variables[key.strip()] = value.strip()
    return variables


def display_execution_plan(steps: list, is_ckp: bool = False, ckp_reader: CKPReader = None, variables: dict = None):
    """Display the execution plan before running."""
    print("\n" + "=" * 80)
    print("EXECUTION PLAN")
    print("=" * 80)
    
    if is_ckp and ckp_reader:
        print(f"Procedure ID: {ckp_reader.get_procedure_id()}")
        print(f"Description: {ckp_reader.get_description()}")
        print(f"Total Steps: {len(steps)}")
        
        if variables:
            print(f"\nVariables ({len(variables)}):")
            for key, value in variables.items():
                # Mask sensitive values
                display_value = value if key not in ["password", "user_id", "secret"] else "***"
                print(f"  • {key} = {display_value}")
        
        # Display rules if any
        rules = ckp_reader.get_rules()
        if rules:
            print(f"\nBusiness Rules ({len(rules)}):")
            for rule in rules:
                print(f"  • {rule.get('rule_id')} - Applies to step {rule.get('applies_to_step')}")
        
        # Display validations if any
        validations = ckp_reader.get_validations()
        if validations:
            print(f"\nValidations ({len(validations)}):")
            for validation in validations:
                print(f"  • {validation.get('validation_id')} - Step {validation.get('step_id')}")
        
        # Display exceptions if any
        exceptions = ckp_reader.get_exceptions()
        if exceptions:
            print(f"\nException Handlers ({len(exceptions)}):")
            for exception in exceptions:
                print(f"  • {exception.get('exception_code')} - {exception.get('severity')}")
        
        # Display recovery flows if any
        recovery_flows = ckp_reader.get_recovery_flows()
        if recovery_flows:
            print(f"\nRecovery Flows ({len(recovery_flows)}):")
            for recovery in recovery_flows:
                print(f"  • {recovery.get('recovery_id')} - For {recovery.get('applies_to_exception')}")
    
    print(f"\nSteps to Execute ({len(steps)}):")
    print("-" * 80)
    
    for i, step in enumerate(steps, 1):
        step_id = step.get("step_id", f"S{i}")
        action = step.get("action", step.get("tool", "unknown"))
        element = step.get("element", "")
        value = step.get("value", "")
        
        # Truncate long values
        if value and len(str(value)) > 50:
            value = str(value)[:47] + "..."
        
        # Format the step display
        print(f"{i:2d}. [{step_id}] {action.upper()}")
        
        if element:
            print(f"     Element: {element}")
        
        if value:
            print(f"     Value: {value}")
        
        # Show additional CKP fields
        if is_ckp:
            window_name = step.get("windowName")
            field_name = step.get("fieldName")
            screen_name = step.get("screenName")
            special_keys = step.get("special_keys")
            
            if window_name:
                print(f"     Window: {window_name}")
            if field_name:
                print(f"     Field: {field_name}")
            if screen_name:
                print(f"     Screen: {screen_name}")
            if special_keys:
                print(f"     Special Keys: {special_keys}")
        
        # Show validation for this step if CKP
        if is_ckp and ckp_reader:
            validation = ckp_reader.get_validation_for_step(step_id)
            if validation:
                expected = validation.get("expected_text_any", [])
                print(f"     ✓ Validation: {validation.get('validation_id')} - Expects: {', '.join(expected)}")
        
        print()
    
    print("=" * 80)
    print()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Mainframe Agent - Execute automation steps via MCP Server with error recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple JSON file
  python main.py example_steps.json

  # CKP file with variables
  python main.py ckp.json --var account_number=123456789 --var account_type=checking

  # CKP file with variables from file
  python main.py ckp.json --vars-file variables.json

  # CKP file with both file and command-line variables (CLI overrides file)
  python main.py ckp.json --vars-file variables.json --var state=CA
        """
    )
    
    parser.add_argument(
        "steps_file",
        type=str,
        help="Path to JSON file containing automation steps (supports CKP format)"
    )
    
    parser.add_argument(
        "--var",
        action="append",
        dest="variables",
        metavar="KEY=VALUE",
        help="Variable in format key=value (can be used multiple times)"
    )
    
    parser.add_argument(
        "--vars-file",
        type=str,
        dest="vars_file",
        help="Path to JSON file containing variables"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="OLLAMA server URL (overrides config)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="OLLAMA model name (overrides config)"
    )
    
    parser.add_argument(
        "--mcp-url",
        type=str,
        default=None,
        help="MCP Server URL (overrides config)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retries per step (overrides config)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip execution plan preview and start immediately"
    )
    
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip delay before execution"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = AgentConfig()
    
    # Override with command-line arguments
    if args.ollama_url:
        config.ollama_url = args.ollama_url
    if args.ollama_model:
        config.ollama_model = args.ollama_model
    if args.mcp_url:
        config.mcp_server_url = args.mcp_url
    if args.max_retries:
        config.max_retries = args.max_retries
    if args.log_level:
        config.log_level = args.log_level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate steps file
    steps_file = Path(args.steps_file)
    if not steps_file.exists():
        logger.error(f"Steps file not found: {args.steps_file}")
        sys.exit(1)
    
    # Load variables
    variables = {}
    
    # Load from file if provided
    if args.vars_file:
        vars_file_path = Path(args.vars_file)
        if not vars_file_path.exists():
            logger.error(f"Variables file not found: {args.vars_file}")
            sys.exit(1)
        variables.update(load_variables_from_file(str(vars_file_path)))
        logger.info(f"Loaded variables from {args.vars_file}")
    
    # Override with command-line variables
    if args.variables:
        cli_vars = parse_variable_args(args.variables)
        variables.update(cli_vars)
        logger.info(f"Loaded {len(cli_vars)} variables from command line")
    
    # Check if it's a CKP file and load steps
    is_ckp = False
    steps = []
    ckp_reader = None
    
    try:
        with open(steps_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "procedure_id" in data and "base_steps" in data:
                is_ckp = True
                # Load CKP file
                ckp_reader = CKPReader(str(steps_file), variables)
                
                # Validate variables
                validation = ckp_reader.validate_variables(variables)
                if not validation["valid"]:
                    logger.error("Variable validation failed:")
                    for error in validation["errors"]:
                        logger.error(f"  • {error}")
                    sys.exit(1)
                
                # Get steps with variable substitution
                steps = ckp_reader.get_steps()
            else:
                # Simple JSON format
                steps = StepReader.load_steps(str(steps_file))
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}", exc_info=True)
        sys.exit(1)
    
    # Display execution plan
    if not args.skip_preview:
        display_execution_plan(steps, is_ckp=is_ckp, ckp_reader=ckp_reader, variables=variables)
        
        # Ask for confirmation (optional - can be made interactive later)
        if not args.no_delay:
            print("Ready to execute. Starting in 2 seconds...")
            time.sleep(2)
        else:
            print("Ready to execute. Starting now...")
    else:
        print(f"Loading {len(steps)} steps... (preview skipped)")
    
    # Initialize and run agent
    async with MainframeAgent(config) as agent:
        try:
            logger.info("=" * 60)
            logger.info("Mainframe Agent - Starting Execution")
            logger.info("=" * 60)
            logger.info(f"Steps file: {args.steps_file}")
            logger.info(f"File type: {'CKP' if is_ckp else 'Simple JSON'}")
            
            if is_ckp:
                logger.info(f"Variables: {len(variables)} provided")
                if variables:
                    logger.info(f"  Variables: {', '.join(variables.keys())}")
            
            logger.info(f"OLLAMA URL: {config.ollama_url}")
            logger.info(f"OLLAMA Model: {config.ollama_model}")
            logger.info(f"MCP Server URL: {config.mcp_server_url}")
            logger.info(f"Max Retries: {config.max_retries}")
            logger.info("=" * 60)
            
            # Execute steps
            if is_ckp:
                result = await agent.execute(steps, ckp_reader=ckp_reader, variables=variables)
            else:
                result = await agent.execute(steps)
            
            # Print summary
            print("\n" + "=" * 60)
            print("EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Success: {result['success']}")
            print(f"Total Steps: {result['total_steps']}")
            print(f"Successful Steps: {result['successful_steps']}")
            print(f"Failed Steps: {result['failed_steps']}")
            print("=" * 60)
            
            # Print detailed results
            if result.get("results"):
                print("\nDETAILED RESULTS:")
                print("-" * 60)
                for i, step_result in enumerate(result["results"], 1):
                    status = "✓" if step_result.get("success") else "✗"
                    step_id = step_result.get("step_id", f"S{i}")
                    action = step_result.get("step", {}).get("action", "unknown")
                    message = step_result.get("message", "No message")
                    print(f"{status} Step {step_id} ({action}): {message}")
                    
                    if not step_result.get("success"):
                        error = step_result.get("error", "Unknown error")
                        print(f"  Error: {error}")
            
            # Exit with appropriate code
            sys.exit(0 if result["success"] else 1)
            
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
