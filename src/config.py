"""
Configuration management module for HYGO project.

This module handles loading and validation of configuration from .env files.
Optimized for minimal memory footprint and fast loading.
"""

from typing import Dict
from pathlib import Path

# Use lazy import to reduce startup time
_dotenv_values = None


def _get_dotenv_values():
    """Lazy load dotenv_values to optimize import time."""
    global _dotenv_values
    if _dotenv_values is None:
        from dotenv import dotenv_values
        _dotenv_values = dotenv_values
    return _dotenv_values


def load_config(env_path: str = ".env") -> Dict[str, str]:
    """
    Load configuration from a local .env file (not OS environment).
    
    Uses dotenv_values for direct file parsing without polluting os.environ.
    Time complexity: O(n) where n is number of lines in .env
    Space complexity: O(k) where k is number of config keys
    
    Args:
        env_path: Path to .env file relative to project root
        
    Returns:
        Dictionary with configuration keys and values
        
    Raises:
        RuntimeError: If required config keys are missing
        FileNotFoundError: If .env file doesn't exist
    """
    dotenv_values = _get_dotenv_values()
    
    # Check if file exists before attempting to load
    if not Path(env_path).exists():
        raise FileNotFoundError(
            f"Configuration file '{env_path}' not found. "
            "Create a .env file with required configuration. "
            "See .env.example for template."
        )
    
    # Load config - O(n) operation
    cfg = dotenv_values(env_path)
    
    # Validate required fields - O(1) lookups
    api_key = cfg.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing in .env. "
            "Set it to your OpenRouter API key."
        )
    
    model = cfg.get("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is missing in .env. "
            "Set it to a vision-capable model name (e.g., openai/gpt-4o-mini)."
        )
    
    # Return validated config with defaults for optional fields
    return {
        "api_key": api_key,
        "model": model,
        "referer": cfg.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "app_title": cfg.get("OPENROUTER_APP_TITLE", "hygo-vision-agent"),
    }


def validate_model_name(model: str) -> bool:
    """
    Basic validation of model name format.
    
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        model: Model name string (e.g., "openai/gpt-4o-mini")
        
    Returns:
        True if format appears valid, False otherwise
    """
    # Model names typically have format "provider/model-name"
    return "/" in model and len(model) > 3


def validate_openrouter_api_key(api_key: str) -> bool:
    """
    Validate OpenRouter API key by calling /api/v1/key endpoint.
    
    Checks if the API key is valid without performing costly operations.
    Uses OpenRouter's key validation endpoint with Bearer token.
    
    Time complexity: O(1) network call
    Space complexity: O(1)
    
    Args:
        api_key: OpenRouter API key to validate
        
    Returns:
        True if key is valid, False otherwise
        
    Raises:
        RuntimeError: If network error occurs or unexpected response
    """
    import requests
    
    # Validate key format first - O(1)
    if not api_key or not api_key.startswith("sk-or-"):
        print("❌ API key format appears invalid (should start with 'sk-or-')")
        return False
    
    url = "https://openrouter.ai/api/v1/key"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    try:
        # Network call with timeout to prevent hanging - O(latency)
        resp = requests.get(url, headers=headers, timeout=10)
        
    except requests.RequestException as e:
        # Network errors are not fatal, but inform user
        print(f"⚠️  Network error checking OpenRouter key: {e}")
        print("   Proceeding anyway, but key validation is skipped.")
        return True  # Soft fail to allow offline development
    
    # Check response status
    if resp.status_code == 200:
        # Key is valid - extract metadata
        try:
            data = resp.json().get("data", {})
            name = data.get("name", "<unnamed>")
            limit = data.get("limit")
            print(f"✅ OpenRouter API key is valid")
            print(f"   Key name: {name}")
            if limit:
                print(f"   Usage limit: {limit}")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not parse key metadata: {e}")
            return True  # Key is still valid, just couldn't parse response
    
    elif resp.status_code == 401:
        # Key is invalid - hard fail
        print("❌ OpenRouter API key is invalid or unauthorized (401)")
        print("   Double-check the OPENROUTER_API_KEY value in your .env file")
        return False
    
    else:
        # Unexpected response
        print(f"⚠️  Unexpected response from OpenRouter: {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        return True  # Soft fail to allow for temporary API issues
