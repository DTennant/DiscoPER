"""
Claude Code Backend - invokes the `claude` CLI as a subprocess.

Each agent call spawns a Claude Code process in print mode (-p) that can
use tools (Bash, Read) to explore data and execute analysis scripts.
Agents are read-only: they can read data files and run Python via Bash,
but cannot write persistent files (Write tool is not granted by default).
"""

import os
import json
import subprocess
import shutil
from typing import Optional


class ClaudeCodeBackend:
    """Backend that invokes the `claude` CLI for agentic tool-use tasks."""

    def __init__(
        self,
        max_turns: int = 15,
        timeout: int = 300,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
    ):
        self.max_turns = max_turns
        self.timeout = timeout
        self.model = model
        self.allowed_tools = allowed_tools or [
            "Bash(command:*)",
            "Read",
            "Glob",
            "Grep",
        ]

        # Verify claude CLI is available
        claude_path = shutil.which("claude")
        if claude_path is None:
            raise RuntimeError(
                "claude CLI not found on PATH. "
                "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _build_env(self) -> dict[str, str]:
        """Build environment for the subprocess, unsetting CLAUDECODE to allow nesting."""
        env = os.environ.copy()
        # Remove CLAUDECODE to avoid "nested session" error
        env.pop("CLAUDECODE", None)
        return env

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Invoke Claude Code in print mode and return its text response.

        Args:
            prompt: The task prompt to send to Claude Code.
            system_prompt: Optional system prompt override.
            working_dir: Working directory for the subprocess.
            max_turns: Override default max_turns.
            timeout: Override default timeout in seconds.

        Returns:
            The text response from Claude Code.
        """
        cmd = ["claude", "-p", "--output-format", "json",
               "--dangerously-skip-permissions"]

        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        turns = max_turns or self.max_turns
        cmd += ["--max-turns", str(turns)]

        if self.model:
            cmd += ["--model", self.model]

        if self.allowed_tools:
            cmd += ["--allowedTools", ",".join(self.allowed_tools)]

        effective_timeout = timeout or self.timeout

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=working_dir,
                env=self._build_env(),
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Claude Code subprocess timed out after {effective_timeout}s"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Claude Code exited with code {result.returncode}: {stderr}"
            )

        # Parse JSON envelope
        raw = result.stdout.strip()
        if not raw:
            raise RuntimeError("Claude Code returned empty output")

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            # If not valid JSON, treat entire output as text
            return raw

        if envelope.get("is_error"):
            raise RuntimeError(
                f"Claude Code returned error: {envelope.get('result', 'unknown')}"
            )

        subtype = envelope.get("subtype", "")
        if subtype == "error_max_turns":
            # Agent ran out of turns before producing final output.
            # The result field may still contain partial text.
            partial = envelope.get("result", "")
            if partial:
                print(f"[CC Backend] Warning: agent hit max_turns ({envelope.get('num_turns')}), "
                      f"using partial result ({len(partial)} chars)")
                return partial
            raise RuntimeError(
                f"Claude Code agent ran out of turns ({envelope.get('num_turns')}) "
                f"with no result text"
            )

        return envelope.get("result", raw)

    def invoke_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_turns: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Invoke Claude Code and parse a JSON object from its response.

        The prompt should instruct Claude Code to output a JSON block.
        This method extracts the LAST ```json ... ``` fenced block
        (since earlier blocks may be intermediate tool output), or falls
        back to raw JSON extraction.

        Returns:
            Parsed dictionary.
        """
        text = self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            working_dir=working_dir,
            max_turns=max_turns,
            timeout=timeout,
        )
        try:
            return self._extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            # Log the raw response for debugging
            print(f"[CC Backend] JSON extraction failed: {e}")
            print(f"[CC Backend] Response length: {len(text)} chars")
            print(f"[CC Backend] Response tail (last 500 chars): ...{text[-500:]}")
            raise

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract a JSON object from text that may contain markdown fences.

        Prefers the LAST fenced ```json block, since Claude Code responses
        often contain intermediate JSON from tool output before the final answer.
        """
        import re

        # Find ALL fenced ```json ... ``` blocks, use the LAST one
        matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
        if matches:
            # Try from last to first
            for match in reversed(matches):
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Find ALL fenced ``` ... ``` blocks (without json tag), use the LAST one
        matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
        if matches:
            for match in reversed(matches):
                candidate = match.group(1).strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

        # Fallback: find the LAST complete JSON object in the text
        # Search from the end for the last '}'
        brace_end = text.rfind("}")
        if brace_end != -1:
            # Walk backwards to find the matching '{'
            depth = 0
            for i in range(brace_end, -1, -1):
                if text[i] == "}":
                    depth += 1
                elif text[i] == "{":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : brace_end + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"Could not extract JSON from Claude Code response:\n{text[:500]}")
