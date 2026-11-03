"""Backend for LLM queries via OpenRouter API (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from core.litellm_backend import ToolSpec

DEFAULT_TIMEOUT = 120


class OpenRouterBackend:
    """Backend using OpenRouter or direct provider APIs via OpenAI SDK."""

    def __init__(self, model_hint: str = ""):
        # Use OpenAI directly for gpt-* models if OPENAI_API_KEY is set
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and ("gpt-" in model_hint or model_hint.startswith("openai/")):
            self._client = OpenAI(
                api_key=openai_key,
                timeout=DEFAULT_TIMEOUT,
                max_retries=3,
            )
            self._provider = "openai"
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=DEFAULT_TIMEOUT,
                max_retries=3,
            )
            self._provider = "openrouter"

    def chat(
        self,
        system: str,
        user: str,
        model: str,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Chat completion returning (text, usage_dict)."""
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

        if model.startswith("openrouter/"):
            model = model[len("openrouter/") :]
        if self._provider == "openai" and model.startswith("openai/"):
            model = model[len("openai/") :]

        create_kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if "max_tokens" in kwargs and "gpt-5" in model:
            create_kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        create_kwargs.update(kwargs)

        completion = self._client.chat.completions.create(**create_kwargs)
        text = completion.choices[0].message.content or ""
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens
            if completion.usage
            else 0,
            "cost": 0.0,
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

        if model.startswith("openrouter/"):
            model = model[len("openrouter/") :]
        if self._provider == "openai" and model.startswith("openai/"):
            model = model[len("openai/") :]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "parameters": tool_spec.output_schema,
                },
            }
        ]

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }
        if "max_tokens" in kwargs and "gpt-5" in model:
            create_kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        create_kwargs.update(kwargs)

        completion = self._client.chat.completions.create(**create_kwargs)
        msg = completion.choices[0].message
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens
            if completion.usage
            else 0,
            "cost": 0.0,
        }
        if msg.tool_calls and len(msg.tool_calls) > 0:
            return json.loads(msg.tool_calls[0].function.arguments), usage
        if msg.content:
            try:
                import re

                match = re.search(r"\{.*\}", msg.content, re.DOTALL)
                if match:
                    return json.loads(match.group()), usage
            except (json.JSONDecodeError, AttributeError):
                pass
        return {}, usage
