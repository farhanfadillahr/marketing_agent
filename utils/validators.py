"""
Validator functions.
"""
from typing import List


def validate_model_type(model_type: str) -> bool:
    """
    Validate model type.
    
    Args:
        model_type: Model type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_models = ["openai", "gemini"]
    return model_type.lower() in valid_models


def validate_agent_type(agent_type: str) -> bool:
    """
    Validate agent type.
    
    Args:
        agent_type: Agent type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_agents = ["general", "marketing"]
    return agent_type.lower() in valid_agents


def get_valid_models() -> List[str]:
    """Get list of valid model types."""
    return ["openai", "gemini"]


def get_valid_agents() -> List[str]:
    """Get list of valid agent types."""
    return ["general", "marketing"]
