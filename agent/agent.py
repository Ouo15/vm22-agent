"""Streaming tool-use loop for the VM-22 agent (Claude API, manual loop)."""

import json

import anthropic

from .prompts import SYSTEM_PROMPT
from .tools import TOOLS, execute

MODEL = "claude-opus-5-5"
BETAS = [
    "server-side-fallback-2026-07-01",      # fallbacks="default": a declined request is retried server-side
    "thinking-display-updates-2026-08-18",  # short progress notes between tool calls
]
MAX_ROUNDS = 25
MAX_JSON_RETRIES = 2


def _blocks_to_send(content):
    """After a mid-output fallback, drop model-internal blocks that precede the last fallback marker."""
    idx = [i for i, b in enumerate(content) if b.type == "fallback"]
    if not idx:
        return list(content)
    cut = idx[-1]
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(content) if i > cut or b.type not in drop]


class VM22Agent:
    def __init__(self, workspace, client=None):
        self.ws = workspace
        self.client = client or anthropic.Anthropic()
        self.messages = []

    def run_turn(self, user_text):
        """Process one user message. Yields UI events as dicts:
        text (delta), status (progress note), tool_start, tool_end, notice, error, done.
        """
        self.messages.append({"role": "user", "content": user_text})
        json_retries = 0
        for _ in range(MAX_ROUNDS):
            try:
                with self.client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=64000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=self.messages,
                    thinking={"type": "adaptive", "display": "updates"},
                    output_config={"effort": "high"},
                    cache_control={"type": "ephemeral"},
                    fallbacks="default",
                    betas=BETAS,
                ) as stream:
                    for event in stream:
                        if event.type == "text":
                            yield {"type": "text", "text": event.text}
                        elif event.type == "content_block_stop" and event.content_block.type == "thinking":
                            note = (event.content_block.thinking or "").strip()
                            if note:
                                yield {"type": "status", "text": note}
                        elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                            yield {"type": "tool_start", "name": event.content_block.name}
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Tool input the SDK could not parse at all: no complete tool_use to answer, so re-issue.
                json_retries += 1
                if json_retries > MAX_JSON_RETRIES:
                    yield {"type": "error", "text": "The model produced unreadable tool input repeatedly."}
                    return
                continue
            except anthropic.AuthenticationError:
                yield {"type": "error", "text": "Authentication failed: check ANTHROPIC_API_KEY."}
                return
            except anthropic.RateLimitError:
                yield {"type": "error", "text": "Rate limited by the API. Wait a moment and try again."}
                return
            except anthropic.APIStatusError as exc:
                yield {"type": "error", "text": f"API error {exc.status_code}: {exc.message}"}
                return
            except anthropic.APIConnectionError:
                yield {"type": "error", "text": "Could not reach the API. Check your connection."}
                return

            if any(b.type == "fallback" for b in response.content):
                yield {"type": "notice", "text": f"Answered by fallback model {response.model}."}

            content = _blocks_to_send(response.content)
            self.messages.append({"role": "assistant", "content": content})

            if response.stop_reason == "refusal":
                yield {"type": "error", "text": "The request was declined. Try rephrasing it."}
                return
            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    yield {"type": "notice", "text": "Response hit the length limit and may be cut off."}
                yield {"type": "done"}
                return
            if response.stop_reason == "max_tokens":
                # A truncated tool input parses as a valid partial object: don't run it.
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Tool input was cut off by the length limit; call the tool again with a shorter input."}
                    for b in tool_uses]})
                continue

            results = []
            for block in tool_uses:
                text, is_error = execute(self.ws, block.name, block.input)
                yield {"type": "tool_end", "name": block.name, "input": block.input,
                       "is_error": is_error, "summary": text[:400]}
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})
            self.messages.append({"role": "user", "content": results})

        yield {"type": "error", "text": f"Stopped after {MAX_ROUNDS} model calls in one turn."}
