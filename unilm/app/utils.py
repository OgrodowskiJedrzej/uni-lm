import yaml
from pathlib import Path
import logging
from typing import Any

logger = logging.getLogger(__name__)

_config: dict[str, Any] | None = None


def load_yaml() -> dict[str, Any]:
    """Load YAML configuration file. Uses global cache if already loaded."""
    global _config
    
    if _config is not None:
        return _config
    
    try:
        path = Path(__file__).parent / "prompts.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path.absolute()}")
        
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        
        if not _config:
            raise ValueError("Config file is empty or invalid YAML")
        
        logger.info(f"Configuration loaded from {path.absolute()}")
        return _config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def select_model(intent: str) -> str:
    """Select the appropriate model based on intent."""
    mapping: dict[str, str] = {
        "coding": "gpt-4o-mini",
        "tests": "gpt-4o-mini",
        "reasoning": "gpt-4o-mini",
        "normal": "gpt-4o-mini",
    }
    
    if intent not in mapping:
        logger.warning(f"Unknown intent '{intent}', defaulting to 'gpt-4o-mini'")
        return "gpt-4o-mini"
    
    return mapping[intent]


def get_agent_prompt(role: str) -> str:
    """Get system prompt for a specific agent role."""
    config = load_yaml()
    
    if 'agents' not in config:
        raise KeyError("'agents' key not found in config")
    
    if role in config['agents']:
        prompt = config['agents'][role].get('system_prompt')
        if not prompt:
            raise ValueError(f"No system_prompt found for role '{role}'")
        return prompt
    
    logger.warning(f"Role '{role}' not found, using 'normal' as fallback")
    return config['agents']['normal']['system_prompt']


def get_orchestrator_prompt() -> str:
    """Get the orchestrator system prompt for intent classification."""
    config = load_yaml()
    
    if 'orchestrator' not in config:
        raise KeyError("'orchestrator' key not found in config")
    
    prompt = config.get('orchestrator', {}).get('system_prompt')
    if not prompt:
        raise ValueError("No system_prompt found in orchestrator config")
    
    return prompt