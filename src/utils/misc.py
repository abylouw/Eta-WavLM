"""
Miscellaneous Utility Functions

Provides general-purpose helper utilities used across the EtaWavLM project.
Includes filesystem and path-related tools such as project root discovery.
"""

import os
from pathlib import Path

from typing import Dict, Any
import yaml


def get_project_root() -> Path:
    """
    Find the project root directory by looking for common project indicators.
    Falls back to script's parent directory if not found.
    """
    current_path = Path(__file__).parent.absolute()
    
    # Look for common project root indicators
    root_indicators = [
        'setup.py', 'pyproject.toml', 'requirements.txt', 
        '.git', 'README.md', 'README.rst', 'environment.yml'
    ]
    
    # Check current directory and parents
    for path in [current_path] + list(current_path.parents):
        for indicator in root_indicators:
            if (path / indicator).exists():
                return path
            
    # If no indicators found, assume the directory containing the script is the project root
    # or if it's clearly named 'eta', use that
    if current_path.name == 'eta':
        return current_path
    elif current_path.parent.name == 'eta':
        return current_path.parent
    else:
        # Create 'eta' directory as project root
        eta_path = current_path / 'eta'
        eta_path.mkdir(exist_ok=True)
        return eta_path

    
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

    
def merge_configs(
        model_config: Dict[str, Any], 
        data_config: Dict[str, Any], 
        train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model, data, and training configurations"""
    # Start with model config
    merged = model_config.copy()
    
    # Add data config fields
    for key, value in data_config.items():
        if key not in merged:
            merged[key] = value
            
    # Apply training config overrides
    overrides = train_config.get('overrides', {})
    for key, value in overrides.items():
        merged[key] = value
        logger.info(f"Applied override: {key} = {value}")
        
    # Add training-specific fields
    training_fields = ['training', 'checkpoint', 'logging', 'evaluation', 'output', 'debug', 'reproducibility']
    for field in training_fields:
        if field in train_config:
            merged[field] = train_config[field]
            
    # Add metadata
    merged['experiment_name'] = train_config.get('experiment_name', 'eta_wavlm_experiment')
    merged['seed'] = train_config.get('seed', 42)
    
    return merged


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training config that references model and data configs"""
    project_root = get_project_root()
    
    # Resolve config_path relative to project root if needed
    if not os.path.isabs(config_path):
        config_path = project_root / config_path
        
    train_config = load_config(str(config_path))
    
    # Load referenced configs (resolve relative to project root)
    model_config_path = project_root / train_config['model_config']
    data_config_path = project_root / train_config['data_config']
    
    model_config = load_config(str(model_config_path))
    data_config = load_config(str(data_config_path))
    
    logger.info(f"Loaded model config: {model_config_path}")
    logger.info(f"Loaded data config: {data_config_path}")
    
    # Merge configurations
    merged = merge_configs(model_config, data_config, train_config)
    return merged


def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load training config that references model and data configs"""
    project_root = get_project_root()
    
    # Resolve config_path relative to project root if needed
    if not os.path.isabs(config_path):
        config_path = project_root / config_path
        
    config = load_config(str(config_path))

    # already merged
    return config
