import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")

_client: openai.OpenAI = None  # type: ignore
_is_custom_api: bool = False  # Track if using custom API

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client, _is_custom_api

    # Support for custom OpenAI-compatible APIs
    custom_base_url = os.environ.get("CUSTOM_BASE_URL")
    custom_api_key = os.environ.get("CUSTOM_API_KEY")

    if custom_base_url and custom_api_key:
        _client = openai.OpenAI(
            api_key=custom_api_key,
            base_url=custom_base_url,
            max_retries=0
        )
        _is_custom_api = True
        logger.info(f"[OpenAI Backend] Using custom API: {custom_base_url}")
    else:
        _client = openai.OpenAI(max_retries=0)
        _is_custom_api = False


def _should_force_tool_choice() -> bool:
    """Determine if tool_choice should be forced.

    Many custom OpenAI-compatible APIs (like CherryIn) don't support
    the tool_choice parameter to force function calling. This function
    checks the environment variable to decide whether to force it.

    Set CUSTOM_FORCE_TOOL_CHOICE=true to enable forced tool_choice.
    Default is False for custom APIs, True for official OpenAI API.
    """
    force_env = os.environ.get("CUSTOM_FORCE_TOOL_CHOICE", "").lower()
    if force_env in ("true", "1", "yes"):
        return True
    if force_env in ("false", "0", "no"):
        return False
    # Default: don't force for custom APIs, force for official OpenAI
    return not _is_custom_api


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Handle function calling setup
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # Only force tool_choice if supported (official OpenAI API or explicitly enabled)
        if _should_force_tool_choice():
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
            logger.debug("[OpenAI Request] Using forced tool_choice")
        else:
            # For custom APIs that don't support tool_choice, we rely on the model
            # to use the tool based on the prompt. Add instruction to system message.
            logger.info("[OpenAI Request] Not forcing tool_choice (custom API mode)")
            # Enhance the system message to encourage tool use
            if system_message:
                tool_instruction = f"\n\nIMPORTANT: You MUST use the '{func_spec.name}' function to respond. Do not respond with plain text."
                system_message = system_message + tool_instruction

    # Build messages after potential system_message modification
    messages = opt_messages_to_list(system_message, user_message)

    # Log request details for debugging
    model_name = filtered_kwargs.get("model", "unknown")
    has_tools = "tools" in filtered_kwargs
    msg_preview = str(messages)[:500] if messages else "None"
    logger.info(f"[OpenAI Request] Model: {model_name}, Has tools: {has_tools}")
    logger.debug(f"[OpenAI Request] Messages preview: {msg_preview}...")

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        # Check if the model used the tool
        if not choice.message.tool_calls:
            # Model didn't use the tool - this can happen with custom APIs that don't support tool_choice
            error_msg = (
                f"Model did not use the required function '{func_spec.name}'. "
                f"Instead returned: {choice.message.content[:200] if choice.message.content else '(empty)'}... "
                f"This may indicate the model doesn't support function calling properly. "
                f"Try setting CUSTOM_FORCE_TOOL_CHOICE=true or use a different model."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if choice.message.tool_calls[0].function.name != func_spec.name:
            raise ValueError(
                f"Function name mismatch: expected '{func_spec.name}', "
                f"got '{choice.message.tool_calls[0].function.name}'"
            )

        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
