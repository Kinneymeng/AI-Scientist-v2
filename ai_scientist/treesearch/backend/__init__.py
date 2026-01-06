import os

from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def _resolve_model_name(model: str, is_vlm_request: bool = False) -> str:
    """Resolve 'custom' model name to actual model from environment variable.

    Args:
        model: The model name from config (e.g., 'custom', 'gpt-4o')
        is_vlm_request: Whether this request contains image content (VLM request)

    Returns:
        The resolved model name to use for the API call.
    """
    if model == "custom" or model.startswith("custom/"):
        # For VLM requests, prefer CUSTOM_VLM_MODEL if set
        if is_vlm_request:
            custom_vlm_model = os.environ.get("CUSTOM_VLM_MODEL")
            if custom_vlm_model:
                return custom_vlm_model
            # Fall back to CUSTOM_MODEL if CUSTOM_VLM_MODEL is not set
            # This will likely fail if CUSTOM_MODEL is not a VLM, but we'll let the API return an error

        custom_model = os.environ.get("CUSTOM_MODEL")
        if not custom_model:
            raise ValueError(
                "Model is 'custom' but CUSTOM_MODEL environment variable is not set. "
                "Please set CUSTOM_MODEL to your actual model name (e.g., 'agent/glm-4.6(free)')."
            )
        return custom_model
    return model


def _is_vlm_request(user_message) -> bool:
    """Check if the user message contains image content (VLM request).

    Args:
        user_message: The user message, which can be a string, dict, or list.

    Returns:
        True if the message contains image_url content, False otherwise.
    """
    if user_message is None:
        return False

    # If it's a list (multi-modal message format)
    if isinstance(user_message, list):
        for item in user_message:
            if isinstance(item, dict):
                # Check for image_url type
                if item.get("type") == "image_url":
                    return True
                # Check for nested content with image_url
                content = item.get("content")
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                            return True

    # If it's a dict with content field
    if isinstance(user_message, dict):
        content = user_message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True

    return False


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """
    # Check if this is a VLM request (contains images)
    is_vlm_request = _is_vlm_request(user_message)

    # Check if using custom model and resolve to actual model name
    is_custom_model = model == "custom" or model.startswith("custom/")
    resolved_model = _resolve_model_name(model, is_vlm_request=is_vlm_request)

    # Log which model is being used for debugging
    if is_vlm_request and is_custom_model:
        import logging
        logger = logging.getLogger("ai-scientist")
        logger.info(f"[VLM Request] Using model: {resolved_model} (detected image content in request)")

    model_kwargs = model_kwargs | {
        "model": resolved_model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if resolved_model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    # Custom models use OpenAI-compatible backend
    if is_custom_model:
        query_func = backend_openai.query
    else:
        query_func = backend_anthropic.query if "claude-" in resolved_model else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
