"""
LiteLLM backend for AIDE agent.

Usage:
    from aide.backend import LLMBackend

    backend = LLMBackend()
    agent = AIDE(
        ...
        generate_chat=backend.chat,
        eval_chat_json=backend.chat_json,
    )

Connects to a LiteLLM proxy server for multi-provider model access and cost tracking.
Set LITELLM_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

from openai import OpenAI
import httpx

# Optional import for AIDE integration
if TYPE_CHECKING:
    from aide.agent import ToolSpec
else:
    try:
        from aide.agent import ToolSpec
    except ImportError:
        ToolSpec = Any  # Fallback if aide not available

# 20 minute timeout for long-running model requests (e.g., gpt-5.1-codex drafts)
DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=1800.0, write=1200.0, pool=1200.0)


# NOTE: reasoning_effort + LiteLLM behavior (summary)
# - This backend just forwards **kwargs (incl. reasoning_effort) to the LiteLLM proxy.
# - Gemini models: reasoning_effort is officially supported; LiteLLM maps it to
#   Gemini "thinking" / thinking_level for you. See:
#   https://docs.litellm.ai/docs/providers/gemini
# - Anthropic/Claude models: reasoning_effort and thinking are supported via
#   LiteLLM’s Anthropic integration (extended thinking / effort for Opus 4.5).
#   See: https://docs.litellm.ai/docs/providers/anthropic
# - OpenAI non-reasoning models (e.g. gpt-4o): OpenAI itself does not support
#   reasoning_effort; LiteLLM will either raise UnsupportedParamsError or drop
#   the param if configured with drop_params=True. See:
#   https://docs.litellm.ai (search: "drop_params" and "reasoning_effort")


class LLMBackend:
    """Backend for LLM queries via LiteLLM proxy server."""

    def __init__(self):
        base_url = (
            os.environ.get("LITELLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://YOUR_API_ENDPOINT/"
        )
        api_key = (
            os.environ.get("LITELLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("LITELLM_API_KEY environment variable not set")
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT, max_retries=3
        )

    def chat(
        self,
        system: str,
        user: str,
        model: str,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Chat completion returning (text, usage_dict with input_tokens, output_tokens, cost).

        Args:
            images: Optional list of base64-encoded PNG strings to include as vision content.
        """
        user_content: str | list[dict[str, Any]]
        if images:
            user_content = [{"type": "text", "text": user}]
            for img_b64 in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
        else:
            user_content = user
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        raw = self._client.chat.completions.with_raw_response.create(
            model=model, messages=messages, **kwargs
        )
        completion = raw.parse()
        text = completion.choices[0].message.content or ""
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens
            if completion.usage
            else 0,
            "cost": float(raw.headers.get("x-litellm-response-cost", 0) or 0),
        }
        return text, usage

    def chat_json(
        self,
        system: str,
        user: str,
        model: str,
        tool_spec: ToolSpec,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Chat completion with function calling returning (parsed_json, usage_dict)."""
        user_content: str | list[dict[str, Any]]
        if images:
            user_content = [{"type": "text", "text": user}]
            for img_b64 in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
        else:
            user_content = user
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        tools = [
            {
                "type": "function",
                "name": tool_spec.name,
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "parameters": tool_spec.output_schema,
                },
            }
        ]
        raw = self._client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            **kwargs,
        )
        completion = raw.parse()
        msg = completion.choices[0].message
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens
            if completion.usage
            else 0,
            "cost": float(raw.headers.get("x-litellm-response-cost", 0) or 0),
        }
        if msg.tool_calls and len(msg.tool_calls) > 0:
            return json.loads(msg.tool_calls[0].function.arguments), usage
        return {}, usage
