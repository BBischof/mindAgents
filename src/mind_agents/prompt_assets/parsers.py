from typing import Optional

from mind_agents.prompt_assets.types import Model, Response


def parse_response_for_model(model: Model, response_text: str) -> Optional[Response]:
    """Parse the response from a model into a structured format.

    Args:
        model: The model that generated the response
        response_text: The raw response text from the model

    Returns:
        A Response object containing the parsed information, or None if parsing fails
    """
    try:
        # For now, all models use the same response format
        # In the future, we can add model-specific parsing if needed

        # Extract the tool call from the response
        # Example response format:
        # I will wait for 5.2 seconds because there might be lower cards.
        # {"tool": "wait_for_n_seconds", "parameters": {"seconds": 5.2, "reason": "..."}}

        # Find the JSON part of the response (after any natural language)
        json_start = response_text.find("{")
        if json_start == -1:
            return None

        # Split into content and JSON parts
        content = response_text[:json_start].strip()
        json_str = response_text[json_start:]

        # Parse the JSON into a Response object
        import json

        tool_call = json.loads(json_str)

        return Response(
            content=content,
            raw_response=response_text,
            success=True,
            tool_calls=[tool_call],
            error=None,
        )

    except Exception as e:
        return Response(
            content=None,
            raw_response=response_text,
            success=False,
            tool_calls=None,
            error=str(e),
        )
