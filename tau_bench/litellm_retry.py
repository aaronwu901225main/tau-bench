# Copyright Sierra

import os
import random
import time
from typing import Any

from litellm import completion


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, parsed)


def _get_non_negative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, parsed)


def is_retryable_litellm_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__.lower()
    message = str(exc).lower()

    non_retryable_keywords = [
        "contextwindowexceeded",
        "maximum context length",
        "content_policy",
        "invalid_api_key",
        "authentication",
        "permission",
        "invalid model",
    ]
    if any(keyword in class_name or keyword in message for keyword in non_retryable_keywords):
        return False

    retryable_keywords = [
        "ratelimit",
        "timeout",
        "apiconnection",
        "serviceunavailable",
        "internalserver",
        "overloaded",
        "temporarily",
        "connection reset",
        "connection aborted",
        "502",
        "503",
        "504",
        "429",
        "could not parse the json body of your request",
    ]
    if any(keyword in class_name or keyword in message for keyword in retryable_keywords):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    return False


def is_header_contamination_error(exc: Exception) -> bool:
    return "unexpected tokens remaining in message header" in str(exc).lower()


def completion_with_retry(**kwargs: Any):
    max_attempts = _get_int_env("TAU_LITELLM_MAX_ATTEMPTS", 5)
    # 0 means no limit for this specific error type.
    header_max_attempts = _get_non_negative_int_env(
        "TAU_LITELLM_HEADER_MAX_ATTEMPTS", 0
    )
    base_backoff = _get_float_env("TAU_LITELLM_RETRY_BACKOFF", 1.5)
    max_backoff = _get_float_env("TAU_LITELLM_RETRY_MAX_BACKOFF", 20.0)

    attempt = 1
    while True:
        try:
            return completion(**kwargs)
        except Exception as exc:
            is_header_error = is_header_contamination_error(exc)
            can_retry = is_header_error or is_retryable_litellm_error(exc)

            effective_max_attempts = (
                header_max_attempts if is_header_error and header_max_attempts > 0 else max_attempts
            )
            max_attempts_reached = (
                False
                if is_header_error and header_max_attempts == 0
                else attempt >= effective_max_attempts
            )

            if (not can_retry) or max_attempts_reached:
                raise

            exponent = min(attempt - 1, 10)
            backoff = min(max_backoff, base_backoff * (2 ** exponent))
            jitter = random.uniform(0, 0.2 * max(1.0, backoff))
            sleep_seconds = backoff + jitter
            limit_display = (
                "inf"
                if is_header_error and header_max_attempts == 0
                else str(effective_max_attempts)
            )
            print(
                f"[LiteLLMRetry] attempt {attempt}/{limit_display} failed with "
                f"{exc.__class__.__name__}: {exc}. retrying in {sleep_seconds:.2f}s..."
            )
            time.sleep(sleep_seconds)
            attempt += 1
