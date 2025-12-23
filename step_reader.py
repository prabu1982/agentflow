"""JSON step file reader and parser."""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class StepReader:
    """Reader for JSON step files."""
    
    @staticmethod
    def load_steps(file_path: str) -> List[Dict[str, Any]]:
        """
        Load steps from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing steps
            
        Returns:
            List of step dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the file doesn't contain a valid steps structure
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Step file not found: {file_path}")
        
        logger.info(f"Loading steps from: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Support multiple JSON structures:
            # 1. Direct array of steps: [{"tool": "...", ...}, ...]
            # 2. Object with "steps" key: {"steps": [{"tool": "...", ...}, ...]}
            # 3. Object with "workflow" key: {"workflow": {"steps": [...]}}
            
            if isinstance(data, list):
                steps = data
            elif isinstance(data, dict):
                if "steps" in data:
                    steps = data["steps"]
                elif "workflow" in data and isinstance(data["workflow"], dict):
                    steps = data["workflow"].get("steps", [])
                else:
                    raise ValueError(
                        "JSON file must contain a 'steps' array or a 'workflow' object with 'steps'"
                    )
            else:
                raise ValueError("JSON file must contain an array or an object with steps")
            
            if not isinstance(steps, list):
                raise ValueError("Steps must be a list/array")
            
            if len(steps) == 0:
                logger.warning("Step file contains no steps")
            
            # Validate each step has required fields
            validated_steps = []
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    logger.warning(f"Skipping invalid step at index {i}: not a dictionary")
                    continue
                
                if "tool" not in step:
                    logger.warning(f"Skipping step at index {i}: missing 'tool' field")
                    continue
                
                validated_steps.append(step)
            
            logger.info(f"Loaded {len(validated_steps)} valid steps from {file_path}")
            return validated_steps
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in step file {file_path}: {str(e)}",
                e.doc,
                e.pos
            )
        except Exception as e:
            logger.error(f"Error loading steps from {file_path}: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def validate_step(step: Dict[str, Any]) -> bool:
        """
        Validate a step structure.
        
        Args:
            step: Step dictionary to validate
            
        Returns:
            True if step is valid, False otherwise
        """
        if not isinstance(step, dict):
            return False
        
        if "tool" not in step:
            return False
        
        return True
