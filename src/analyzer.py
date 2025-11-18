"""
LLM client and analysis logic for HYGO project.

Handles OpenRouter API communication and vision-based error detection.
Optimized for API efficiency and response parsing reliability.
"""

import json
from typing import Any, Dict

from openai import OpenAI
from PIL import Image

from .image_utils import encode_image_to_data_url


# Price-ordered chain of vision-capable models (cheapest → most expensive)
# Based on OpenRouter pricing and confirmed multimodal support
# Note: gpt-5-nano may not support vision - using proven models
CHEAP_VISION_MODELS = [
    "openai/gpt-4o-mini",     # Cost-effective vision ($0.15/1M tokens)
    "google/gemini-flash-1.5",  # Fast and cheap vision alternative
    "anthropic/claude-3-haiku",  # Reliable vision fallback
    "openai/gpt-4o",          # Robust but expensive fallback
]


# Default system prompt for error detection
# Structured for clear, objective analysis with JSON output
DEFAULT_SYSTEM_PROMPT = """You are an expert at spotting visual mistakes in AI-generated images.
You will see a small crop from a larger image.
Look ONLY at the visible region, not at any imagined context.

Identify clear, objective visual errors such as:
- Anatomical impossibilities or deformities (extra/missing limbs, distorted hands, impossible faces, etc.)
- Physically impossible geometry or perspective
- Nonsensical or unreadable text
- Obvious rendering artifacts or glitches

Return a JSON object with exactly two keys:
- issue_count: integer number of distinct clear errors in this region.
- issues: array of short English strings, each describing one error.

If there are no obvious errors, respond with {"issue_count": 0, "issues": []}.
Be conservative - only flag clear, unambiguous errors."""


def _build_model_chain(preferred: str | None = None) -> tuple:
    """
    Build a cheap-first model fallback chain.
    
    If preferred is in the known models list, move it to front.
    Otherwise, use default price-ordered chain.
    
    Time complexity: O(n) where n is length of model chain
    Space complexity: O(n) for chain copy
    
    Args:
        preferred: Optional preferred model name from config
        
    Returns:
        Tuple of (primary_model, fallback_list)
    """
    chain = CHEAP_VISION_MODELS.copy()
    
    # If preferred model is in our chain, prioritize it
    if preferred and preferred in chain:
        chain.remove(preferred)
        chain.insert(0, preferred)
    
    primary = chain[0]
    fallbacks = chain[1:] if len(chain) > 1 else []
    
    return primary, fallbacks


def create_openrouter_client(api_key: str, referer: str, app_title: str) -> OpenAI:
    """
    Create OpenAI client configured for OpenRouter.
    
    Uses OpenAI SDK with custom base_url to route to OpenRouter API.
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        api_key: OpenRouter API key
        referer: HTTP referer for request tracking
        app_title: Application title for OpenRouter dashboard
        
    Returns:
        Configured OpenAI client instance
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": app_title,
        },
    )


def analyze_region(
    client: OpenAI,
    model: str,
    region_img: Image.Image,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    Analyze single image region using vision LLM with automatic fallbacks.
    
    Sends region as base64-encoded data URL with structured prompt.
    Uses JSON mode for reliable parsing of structured responses.
    Implements OpenRouter fallback chain for robust model routing.
    
    Time complexity: O(API_latency + encoding_time)
    Space complexity: O(image_size_encoded)
    
    Args:
        client: OpenAI client configured for OpenRouter
        model: Model name (e.g., "openai/gpt-4o-mini")
        region_img: PIL Image of region to analyze
        system_prompt: Prompt instructing analysis format
        
    Returns:
        Dict with 'issue_count' (int) and 'issues' (list of str)
    """
    # Encode image to data URL - O(w*h) for image size
    data_url = encode_image_to_data_url(region_img, fmt="PNG")
    
    # Build fallback chain with preferred model first
    primary_model, fallback_models = _build_model_chain(model)
    
    try:
        # Call API with vision input, JSON mode, and fallback chain
        # extra_body enables OpenRouter's automatic model fallback
        # If primary model fails/doesn't support vision, fallbacks are tried
        response = client.chat.completions.create(
            model=primary_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,  # Limit response size for cost efficiency
            extra_body={
                "models": fallback_models,  # OpenRouter fallback chain
            },
        )
        
        # Extract response text - O(1) access
        text = response.choices[0].message.content
        
        # Check for empty responses
        if text is None or text.strip() == "":
            print(f"WARNING: Empty response from model. Treating as no-issue.")
            return {"issue_count": 0, "issues": []}
        
    except Exception as e:
        # API errors should not crash pipeline
        print(f"WARNING: API call failed: {e}. Treating as no-issue.")
        return {"issue_count": 0, "issues": []}
    
    # Parse JSON response - O(response_length)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse JSON response: {e}. Treating as no-issue.")
        return {"issue_count": 0, "issues": []}
    
    # Normalize and validate fields - O(1) + O(k) for k issues
    issue_count = int(obj.get("issue_count", 0))
    issues = obj.get("issues", [])
    
    # Ensure issues is a list of strings
    if not isinstance(issues, list):
        issues = [str(issues)]
    else:
        issues = [str(issue) for issue in issues]
    
    # Clamp issue_count to non-negative
    issue_count = max(issue_count, 0)
    
    return {
        "issue_count": issue_count,
        "issues": issues,
    }


def compute_severity(issue_count: int) -> str:
    """
    Map issue count to severity category.
    
    Categorization logic from strategy.md:
    - 0 issues: none (no marker)
    - 1-2 issues: mild (yellow marker)
    - 3+ issues: severe (red marker)
    
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        issue_count: Number of detected issues
        
    Returns:
        Severity string: "none", "mild", or "severe"
    """
    if issue_count <= 0:
        return "none"
    elif 1 <= issue_count <= 2:
        return "mild"
    else:  # 3 or more
        return "severe"


def analyze_image_grid(
    client: OpenAI,
    model: str,
    regions: list,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[tuple, Dict[str, Any]]:
    """
    Analyze all regions from split image grid.
    
    Processes each cell independently and aggregates results.
    Time complexity: O(n * API_latency) for n cells
    Space complexity: O(n) for results storage
    
    Args:
        client: OpenAI client configured for OpenRouter
        model: Model name for analysis
        regions: List of (row, col, image) tuples from split_into_grid
        system_prompt: Analysis instruction prompt
        
    Returns:
        Dict mapping (row, col) to analysis results
    """
    results = {}
    
    # Process each region - O(n) iterations
    for row, col, crop in regions:
        result = analyze_region(client, model, crop, system_prompt)
        results[(row, col)] = result
    
    return results
