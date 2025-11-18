"""
Vision model pricing module for HYGO project.

Fetches and analyzes OpenRouter models with image input capabilities.
Provides pricing information and sorting for cost-effective model selection.
Optimized for API efficiency and data extraction.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import requests

from .config import load_config


# OpenRouter API endpoints
MODELS_URL_USER = "https://openrouter.ai/api/v1/models/user"
MODELS_URL_ALL = "https://openrouter.ai/api/v1/models"


def fetch_all_models(use_user_filter: bool = True, env_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Fetch all models from OpenRouter API.
    
    Uses /models/user to get models filtered by user preferences,
    or /models to get the complete catalog.
    
    Time complexity: O(1) API call + O(n) response parsing
    Space complexity: O(n) for model list
    
    Args:
        use_user_filter: If True, use /models/user endpoint (user-specific)
                        If False, use /models endpoint (full catalog)
    
    Returns:
        List of model dictionaries from API response
        
    Raises:
        RuntimeError: If API call fails or response is invalid
    """
    # Allow caller to pass explicit .env path; fall back to default behavior
    cfg = load_config(env_path=env_path or ".env")
    api_key = cfg["api_key"]
    
    # Select endpoint based on filter preference
    url = MODELS_URL_USER if use_user_filter else MODELS_URL_ALL
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch models from OpenRouter: {e}")
    
    data = resp.json().get("data", [])
    if not isinstance(data, list):
        raise RuntimeError("Unexpected /models response structure - 'data' is not a list")
    
    return data


def filter_vision_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter models to only those supporting image input.
    
    Checks architecture.input_modalities for "image" capability.
    Time complexity: O(n) where n is number of models
    Space complexity: O(m) where m is number of vision models
    
    Args:
        models: List of model dictionaries from API
        
    Returns:
        Filtered list containing only vision-capable models
    """
    vision = []
    for m in models:
        # Get architecture object - O(1) lookup
        arch = m.get("architecture") or {}
        input_mods = arch.get("input_modalities") or []
        
        # Check if image modality is supported - O(k) where k is small
        if "image" in input_mods:
            vision.append(m)
    
    return vision


def extract_pricing(m: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract and normalize pricing fields from model object.
    
    Converts string prices to floats, handles missing values as 0.0.
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        m: Model dictionary from API
        
    Returns:
        Dictionary with float pricing values:
        - prompt: Price per prompt token (USD)
        - completion: Price per completion token (USD)
        - image: Price per image input (USD)
        - request: Price per request (USD)
    """
    p = m.get("pricing") or {}
    
    def to_float(val: Any) -> float:
        """Convert API value to float, handling None/empty strings."""
        if val is None or val == "":
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    
    return {
        "prompt": to_float(p.get("prompt")),
        "completion": to_float(p.get("completion")),
        "image": to_float(p.get("image")),
        "request": to_float(p.get("request")),
    }


def build_vision_price_table(use_user_filter: bool = True, env_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Build sorted table of vision models with pricing information.
    
    Fetches models, filters for vision support, extracts pricing,
    and sorts by image price (ascending - cheapest first).
    
    Time complexity: O(n log n) for sorting where n is number of vision models
    Space complexity: O(n) for result list
    
    Args:
        use_user_filter: If True, use user-specific model list
        
    Returns:
        List of dictionaries, each containing:
        - id: Model identifier (e.g., "openai/gpt-4o-mini")
        - name: Human-readable model name
        - image_price: Price per image (USD)
        - prompt_price: Price per prompt token (USD)
        - completion_price: Price per completion token (USD)
        - context_length: Maximum context window
        
    Sorted ascending by image_price, then prompt_price as tiebreaker.
    """
    # Fetch and filter models - O(n)
    all_models = fetch_all_models(use_user_filter, env_path)
    vision_models = filter_vision_models(all_models)
    
    # Extract relevant fields - O(n)
    rows = []
    for m in vision_models:
        pricing = extract_pricing(m)
        
        # Get context length with safe fallback
        context_len = m.get("context_length", 0)
        if context_len is None:
            context_len = 0
        
        rows.append({
            "id": m.get("id", "unknown"),
            "name": m.get("name", "Unknown Model"),
            "image_price": pricing["image"],
            "prompt_price": pricing["prompt"],
            "completion_price": pricing["completion"],
            "request_price": pricing["request"],
            "context_length": context_len,
        })
    
    # Sort by image price (ascending), then prompt price - O(n log n)
    rows.sort(key=lambda r: (r["image_price"], r["prompt_price"]))
    
    return rows


def print_vision_price_table(use_user_filter: bool = True, env_path: str | None = None) -> None:
    """
    Print formatted table of vision models with pricing to console.
    
    Time complexity: O(n log n) from build_vision_price_table
    Space complexity: O(n)
    
    Args:
        use_user_filter: If True, show only user-accessible models
    """
    rows = build_vision_price_table(use_user_filter, env_path)
    count = len(rows)
    
    print(f"\n{'='*100}")
    print(f"OPENROUTER VISION MODELS - PRICING ANALYSIS")
    print(f"{'='*100}")
    print(f"\nFound {count} models that support image input.\n")
    
    # Table header
    print(
        f"{'Index':>5}  {'Model ID':<40}  {'Image $':>12}  "
        f"{'Prompt $/tok':>14}  {'Compl $/tok':>14}  {'Context':>10}"
    )
    print("-" * 100)
    
    # Table rows
    for idx, r in enumerate(rows, start=1):
        print(
            f"{idx:5d}  {r['id']:<40}  "
            f"{r['image_price']:12.10f}  "
            f"{r['prompt_price']:14.12f}  "
            f"{r['completion_price']:14.12f}  "
            f"{r['context_length']:10,d}"
        )
    
    print("\n" + "="*100 + "\n")


def get_cheapest_vision_models(n: int = 5, use_user_filter: bool = True, env_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Get the N cheapest vision models by image price.
    
    Time complexity: O(n log n) for sorting
    Space complexity: O(n)
    
    Args:
        n: Number of models to return
        use_user_filter: If True, use user-specific model list
        
    Returns:
        List of N cheapest models (or fewer if less available)
    """
    rows = build_vision_price_table(use_user_filter, env_path)
    return rows[:n]


def get_model_pricing(model_id: str, use_user_filter: bool = True, env_path: str | None = None) -> Optional[Dict[str, Any]]:
    """
    Get pricing information for a specific model by ID.
    
    Time complexity: O(n) for searching
    Space complexity: O(n) for table
    
    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o-mini")
        use_user_filter: If True, search user-specific models only
        
    Returns:
        Model pricing dictionary if found, None otherwise
    """
    rows = build_vision_price_table(use_user_filter, env_path)
    
    for row in rows:
        if row["id"] == model_id:
            return row
    
    return None


if __name__ == "__main__":
    # CLI usage: python -m src.vision_pricing
    print_vision_price_table()
