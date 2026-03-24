"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_tokens: int = 8192
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_tokens: Maximum tokens to generate (default 8192)

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
    }

    # Dynamic max_tokens based on model constraints
    # User requested removal of limits
    # if "gpt-5" in model and max_tokens > 4096:
    #     max_tokens = 4096 # Constraint for GPT-5 Pro to avoid 402 under current plan
    # elif "claude-opus" in model and max_tokens > 4096:
    #     max_tokens = 4096 # Safe limit for high-cost model
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens, 
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            content = message.get('content') or ""
            reasoning = message.get('reasoning') or message.get('reasoning_details') or ""
            
            # If content is empty but reasoning exists, use reasoning
            if not content and reasoning:
                content = f"*Reasoning only:*\n\n{reasoning}"
            elif reasoning:
                # Append reasoning if desired, or keep separate. 
                # For now, let's just make sure content isn't empty.
                pass

            return {
                'content': content,
                'reasoning_details': reasoning
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
