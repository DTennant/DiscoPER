# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import os

# Try to import backends
try:
    from .litellm_backend import LLMBackend

    USE_LITELLM = True
except ImportError:
    USE_LITELLM = False

try:
    from .openrouter_backend import OpenRouterBackend

    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False

from .llm_client import LLMClient

OPENROUTER_PROVIDERS = (
    "openai/",
    "google/",
    "anthropic/",
    "deepseek/",
    "moonshotai/",
    "meta/",
    "mistralai/",
)


class Agent:
    def __init__(
        self,
        model_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        log_llm_metrics=False,
        secrets: Optional[dict[str:str]] = None,
        api_version: Optional[str] = None,
    ):
        self.model_name = model_name or "gpt-4"
        self.system_prompt = system_prompt or ""
        self.log_llm_metrics = log_llm_metrics

        use_openrouter = (
            HAS_OPENROUTER
            and (
                os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            )
            and any(
                self.model_name.startswith(p)
                or self.model_name.startswith("openrouter/" + p)
                for p in OPENROUTER_PROVIDERS
            )
        )

        if use_openrouter:
            self.backend = OpenRouterBackend(model_hint=self.model_name)
            self.llm = None
        elif USE_LITELLM:
            self.backend = LLMBackend()
            self.llm = None
        else:
            # Fall back to old LLMClient
            api_key = None
            if secrets:
                for k, v in secrets.items():
                    if k.endswith("OPENAI_API_KEY") and "gemini" not in model_name:
                        api_key = v
                        break
                    if "gemini" in model_name and k.endswith("GEMINI_API_KEY"):
                        api_key = v
                        os.environ["GEMINI_API_KEY"] = api_key
                        break

            self.llm = LLMClient(
                model_url=model_url,
                model_name=model_name,
                log_metrics=log_llm_metrics,
                api_key=api_key,
                api_version=api_version,
            )
            self.backend = None

    def act(
        self,
        instruction: str,
        validator: Optional[Callable[str, Optional[str]]] = None,  # type: ignore
        max_retries=1,
        images: Optional[list[str]] = None,
    ) -> str:
        if self.backend:
            response, usage = self.backend.chat(
                system=self.system_prompt,
                user=instruction,
                model=self.model_name,
                images=images,
            )
        else:
            response = self.llm.generate(instruction)

        if validator:
            response = validator(response)

            n_retries = 0
            while n_retries < max_retries and response is None:
                if self.backend:
                    response, usage = self.backend.chat(
                        system=self.system_prompt,
                        user=instruction,
                        model=self.model_name,
                        images=images,
                    )
                else:
                    response = self.llm.generate(
                        instruction, system_prompt=self.system_prompt
                    )

                n_retries += 1

                response = validator(response)

        if response is None:
            raise ValueError(f"Malformed response after {max_retries} attempts.")

        return response

    def act_json(
        self,
        instruction: str,
        output_schema: dict,
        tool_name: str = "generate_output",
        tool_description: str = "Generate structured output",
        max_retries: int = 1,
        images: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate structured JSON output using function calling.

        Args:
            instruction: The prompt/instruction for the LLM
            output_schema: JSON schema for the expected output
            tool_name: Name of the tool/function
            tool_description: Description of what the tool does
            max_retries: Maximum number of retry attempts
            images: Optional list of base64-encoded PNG strings for vision

        Returns:
            Parsed JSON dictionary
        """
        if not self.backend:
            raise NotImplementedError("act_json only works with LiteLLM backend")

        # Create ToolSpec-like object
        class ToolSpec:
            def __init__(self, name, description, schema):
                self.name = name
                self.description = description
                self.output_schema = schema

        tool_spec = ToolSpec(tool_name, tool_description, output_schema)

        for attempt in range(max_retries + 1):
            try:
                result, usage = self.backend.chat_json(
                    system=self.system_prompt,
                    user=instruction,
                    model=self.model_name,
                    tool_spec=tool_spec,
                    images=images,
                )

                if result:
                    return result
                else:
                    if attempt < max_retries:
                        print(
                            f"Empty response, retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        raise ValueError("No response from LLM after all retries")

            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"Error: {e}, retrying... (attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                else:
                    raise ValueError(
                        f"Failed to get JSON response after {max_retries + 1} attempts: {e}"
                    )

        raise ValueError(
            f"Failed to get JSON response after {max_retries + 1} attempts"
        )

    def flush_logs(self, path: str):
        if self.llm:
            self.llm.flush_logs(path)
