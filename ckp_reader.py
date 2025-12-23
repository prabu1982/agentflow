"""CKP (Checkpoint) JSON file reader and parser with support for variables, rules, validations, and recovery flows."""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CKPReader:
    """Reader for CKP JSON files with full procedure support."""
    
    def __init__(self, file_path: str, variables: Optional[Dict[str, Any]] = None):
        """
        Initialize CKP reader.
        
        Args:
            file_path: Path to CKP JSON file
            variables: Variables to substitute in steps
        """
        self.file_path = Path(file_path)
        self.variables = variables or {}
        self.ckp_data = None
        self._load_file()
    
    def _load_file(self):
        """Load and parse CKP JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CKP file not found: {self.file_path}")
        
        logger.info(f"Loading CKP file: {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.ckp_data = json.load(f)
            
            logger.info(f"Loaded CKP procedure: {self.ckp_data.get('procedure_id', 'unknown')}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in CKP file {self.file_path}: {str(e)}",
                e.doc,
                e.pos
            )
        except Exception as e:
            logger.error(f"Error loading CKP file {self.file_path}: {str(e)}", exc_info=True)
            raise
    
    def get_procedure_id(self) -> str:
        """Get procedure ID."""
        return self.ckp_data.get("procedure_id", "unknown")
    
    def get_description(self) -> str:
        """Get procedure description."""
        return self.ckp_data.get("description", "")
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Get steps with variable substitution.
        
        Returns:
            List of steps with variables substituted
        """
        base_steps = self.ckp_data.get("base_steps", [])
        
        # Substitute variables in steps
        processed_steps = []
        for step in base_steps:
            processed_step = self._substitute_variables(step.copy())
            processed_steps.append(processed_step)
        
        return processed_steps
    
    def _substitute_variables(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute {{variable}} placeholders in step values."""
        if "value" in step and isinstance(step["value"], str):
            # Find all {{variable}} patterns
            pattern = r'\{\{(\w+)\}\}'
            matches = re.findall(pattern, step["value"])
            
            for var_name in matches:
                if var_name in self.variables:
                    value = str(self.variables[var_name])
                    step["value"] = step["value"].replace(f"{{{{{var_name}}}}}", value)
                else:
                    # Check if it's a required variable
                    required_vars = self.ckp_data.get("variables_schema", {}).get("required", {})
                    if var_name in required_vars:
                        raise ValueError(
                            f"Required variable '{var_name}' not provided in step {step.get('step_id')}"
                        )
                    # Optional variable - use default if available
                    optional_vars = self.ckp_data.get("variables_schema", {}).get("optional", {})
                    if var_name in optional_vars:
                        default = optional_vars[var_name].get("default")
                        if default is not None:
                            step["value"] = step["value"].replace(f"{{{{{var_name}}}}}", str(default))
                        else:
                            logger.warning(
                                f"Optional variable '{var_name}' not provided and no default in step {step.get('step_id')}"
                            )
        
        return step
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get business rules."""
        return self.ckp_data.get("rules", [])
    
    def get_validations(self) -> List[Dict[str, Any]]:
        """Get validation definitions."""
        return self.ckp_data.get("validations", [])
    
    def get_exceptions(self) -> List[Dict[str, Any]]:
        """Get exception definitions."""
        return self.ckp_data.get("exceptions", [])
    
    def get_recovery_flows(self) -> List[Dict[str, Any]]:
        """Get recovery flow definitions."""
        return self.ckp_data.get("recovery_flows", [])
    
    def get_variables_schema(self) -> Dict[str, Any]:
        """Get variables schema."""
        return self.ckp_data.get("variables_schema", {})
    
    def validate_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate variables against schema.
        
        Args:
            variables: Variables to validate
            
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        schema = self.get_variables_schema()
        errors = []
        
        # Check required variables
        required = schema.get("required", {})
        for var_name, var_schema in required.items():
            if var_name not in variables:
                errors.append(f"Required variable '{var_name}' is missing")
            else:
                # Validate type and pattern
                var_type = var_schema.get("type")
                pattern = var_schema.get("pattern")
                
                if var_type == "string" and not isinstance(variables[var_name], str):
                    errors.append(f"Variable '{var_name}' must be a string")
                
                if pattern:
                    import re
                    if not re.match(pattern, str(variables[var_name])):
                        errors.append(
                            f"Variable '{var_name}' does not match pattern: {pattern}"
                        )
        
        # Check optional variables
        optional = schema.get("optional", {})
        for var_name, var_schema in optional.items():
            if var_name in variables:
                var_type = var_schema.get("type")
                pattern = var_schema.get("pattern")
                enum = var_schema.get("enum")
                
                if var_type == "string" and not isinstance(variables[var_name], str):
                    errors.append(f"Variable '{var_name}' must be a string")
                
                if pattern:
                    import re
                    if not re.match(pattern, str(variables[var_name])):
                        errors.append(
                            f"Variable '{var_name}' does not match pattern: {pattern}"
                        )
                
                if enum and variables[var_name] not in enum:
                    errors.append(
                        f"Variable '{var_name}' must be one of: {enum}"
                    )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def get_rule_for_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get business rule that applies to a step."""
        rules = self.get_rules()
        for rule in rules:
            if rule.get("applies_to_step") == step_id:
                return rule
        return None
    
    def get_validation_for_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get validation that applies to a step."""
        validations = self.get_validations()
        for validation in validations:
            if validation.get("step_id") == step_id:
                return validation
        return None
    
    def get_recovery_for_exception(self, exception_code: str) -> Optional[Dict[str, Any]]:
        """Get recovery flow for an exception."""
        recovery_flows = self.get_recovery_flows()
        for recovery in recovery_flows:
            if recovery.get("applies_to_exception") == exception_code:
                return recovery
        return None
    
    def get_exception_by_code(self, exception_code: str) -> Optional[Dict[str, Any]]:
        """Get exception definition by code."""
        exceptions = self.get_exceptions()
        for exception in exceptions:
            if exception.get("exception_code") == exception_code:
                return exception
        return None
