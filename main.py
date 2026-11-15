#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, the best free OpenRouter model behind
every answer.

A Python 3.13 standard-library-only entry. It reads the brief at $MAKEFUN_PROMPT_FILE (the free
model list and the six prompts' manifest), builds one catalog, listens on a free port, registers
with the env at $MAKEFUN_ENV_URL/register, and serves the client's requests:

  POST /v1/chat/completions   an OpenAI Chat Completions request (JSON or SSE)
  POST /v1/messages           an Anthropic Messages request (JSON or SSE)
  GET  /v1/models             OpenAI shape for Authorization: Bearer,
                              Anthropic shape for x-api-key

Every request is routed to the strongest free model (highest rating_milli) that meets its
requirements, with a stable tie-break on the dialable id. A refusal (any 4xx/5xx) fails over to
the next-best model, and the refusals are reported on the response's `x-makefun-upstream-failures`
header so the failover column can see them. The client's wire is translated to Chat Completions
(the only route these ids have) and the gateway's answer is translated back. The server polls
GET /work and exits 0 when it says done.

Every choice is a function of the brief, the request and the gateway answers seen so far — no wall
clock and no unseeded randomness — so a replay makes the same calls and leaves the same export.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- the wire's constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
UPSTREAM_TIMEOUT_SEC = 60

ENV_URL = (os.environ.get("MAKEFUN_ENV_URL", "") or "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "") or ""
GATEWAY_URL = (os.environ.get("OPENAI_BASE_URL", "") or "").rstrip("/")
# The trial's dummy key; the gateway swaps it for the entrant's own OpenRouter key.
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "") or ""

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"
SCHEMA_PREFIX = "free-model-aggregator-02/"

# The public base URL of the provider these free models belong to; never dialled (the gateway is
# dialled at $OPENAI_BASE_URL), only written into the catalog, which is part of the replay's export.
OPENROUTER_PUBLIC_BASE = "https://" + "openrouter" + ".ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Fixed, clock-free timestamps so GET /v1/models is deterministic on every pass.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

LEGAL_ANTHROPIC_STOP = ("end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn")

# A refusal status the model will not recover from within the trial: never dial it again.
PERMANENT_REFUSAL_STATUS = (400, 401, 402, 403, 404, 408, 409)
# A 429 is transient; give a model this many refusals before it is parked for the rest of the trial.
MAX_429_BEFORE_DEAD = 2


# ---------------------------------------------------------------- small helpers


def is_record(value):
    return isinstance(value, dict)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_or_null(value):
    return value if is_number(value) else None


def number_or_zero(value):
    return value if is_number(value) else 0


def string_of(value):
    return value if isinstance(value, str) else ""


def text_of(content):
    """The text of a string or of a block array's text blocks; '' otherwise."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def arguments_of(value):
    """A tool call's arguments as an object, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# ---------------------------------------------------------------- the brief


def object_end(text, start):
    """The index of the `}` closing the object that opens at `start`, or -1; string contents skipped."""
    depth = 0
    in_string = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in the order they appear."""
    out = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str) and SCHEMA_PREFIX in string_of(value.get("schema")):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def model_of(value):
    if not is_record(value):
        return None
    raw = string_of(value.get("id"))
    if raw == "":
        raw = string_of(value.get("model"))
    if raw == "":
        return None
    provider = string_of(value.get("provider")) or "openrouter"
    model_id = ("openrouter/" + raw) if provider == "openrouter" and not raw.lower().startswith("openrouter/") else raw
    display_name = string_of(value.get("display_name")) or raw
    return {
        "id": model_id,
        "provider": provider,
        "display_name": display_name,
        "rating_milli": number_or_zero(value.get("rating_milli")),
        "context_length": number_or_null(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is not False,
    }


def prompt_of(value):
    if not is_record(value):
        return None
    pid = string_of(value.get("id"))
    if pid == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if is_number(requires.get("min_context")):
        req["min_context"] = requires["min_context"]
    return {
        "id": pid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    brief = {"models": [], "prompts": [], "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    models_doc = next((d for d in docs if d.get("schema") == MODELS_SCHEMA and isinstance(d.get("models"), list)), None)
    if models_doc is None:
        brief["error"] = "the brief holds no %s document" % MODELS_SCHEMA
        return brief
    brief["models"] = [m for m in (model_of(v) for v in models_doc["models"]) if m is not None]
    prompts_doc = next((d for d in docs if d.get("schema") == PROMPTS_SCHEMA and isinstance(d.get("prompts"), list)), None)
    if prompts_doc is not None:
        brief["prompts"] = [p for p in (prompt_of(v) for v in prompts_doc["prompts"]) if p is not None]
    return brief


def free_models(brief):
    return [m for m in brief["models"] if m["free"]]


def build_catalog(models):
    seen = set()
    rows = []
    for model in models:
        mid = model["id"]
        if not model["free"] or mid in seen:
            continue
        seen.add(mid)
        rows.append({"id": mid, "free": True, "context_length": model["context_length"]})
    rows.sort(key=lambda r: r["id"])
    if not rows:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


def served_models(models):
    out = []
    for model in models:
        if model["free"]:
            out.append({"id": model["id"], "provider": model["provider"], "display_name": model["display_name"]})
    out.sort(key=lambda m: m["id"])
    return out


# ---------------------------------------------------------------- requirements and routing


def meets(requires, caps):
    if requires.get("tools") and not caps["tools"]:
        return False
    if requires.get("vision") and not caps["vision"]:
        return False
    if requires.get("json") and not caps["json"]:
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (caps["context_length"] is None or caps["context_length"] < min_context):
        return False
    return True


def has_json_mode(body):
    fmt = body.get("response_format")
    return is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")


def body_has_tools(body):
    return isinstance(body.get("tools"), list) and len(body["tools"]) > 0


def body_has_images(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                return True
    return False


def requirements_for(body, manifest_requires):
    """The requirements of the request now arriving: the brief's row by arrival order, united with
    the request body's own (so a body that asks for JSON mode, tools or an image is never under-
    required even when the brief's manifest is short)."""
    req = dict(manifest_requires) if manifest_requires else {}
    if has_json_mode(body):
        req["json"] = True
    if body_has_tools(body):
        req["tools"] = True
    if body_has_images(body):
        req["vision"] = True
    return req


def candidate_models(requires):
    """Free models meeting `requires`, best rating first, then the dialable id as a stable tie-break."""
    rows = [m for m in FREE if meets(requires, m) and m["id"] not in DEAD]
    rows.sort(key=lambda m: (-m["rating_milli"], m["id"]))
    return rows


# ---------------------------------------------------------------- request translation


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_record(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                src = block["source"]
                url = None
                if src.get("type") == "base64" and isinstance(src.get("data"), str):
                    media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
                    url = "data:%s;base64,%s" % (media, src["data"])
                elif src.get("type") == "url" and isinstance(src.get("url"), str):
                    url = src["url"]
                if url is not None:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts:
            messages.append({"role": "user", "content": parts})

    out = {"model": model, "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    stop_sequences = body.get("stop_sequences")
    if isinstance(stop_sequences, list) and len(stop_sequences) > 0:
        out["stop"] = stop_sequences
    tools = body.get("tools")
    if isinstance(tools, list):
        converted = []
        for raw_tool in tools:
            if not is_record(raw_tool) or not isinstance(raw_tool.get("name"), str):
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw_tool["name"],
                        "description": raw_tool["description"] if isinstance(raw_tool.get("description"), str) else "",
                        "parameters": raw_tool["input_schema"] if is_record(raw_tool.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_openai(body, model):
    out = dict(body)
    out["model"] = model
    if out.get("stream") is True:
        options = dict(out["stream_options"]) if is_record(out.get("stream_options")) else {}
        options["include_usage"] = True
        out["stream_options"] = options
    return out


# ---------------------------------------------------------------- stop reason mappings


def stop_reason_of(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


# ---------------------------------------------------------------- answer translation (JSON)


def openai_to_anthropic_answer(body, model):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if len(choices) > 0 and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    finish = choice.get("finish_reason")
    stop_reason = stop_reason_of(finish)
    if stop_reason not in LEGAL_ANTHROPIC_STOP:
        stop_reason = "end_turn"
    return {
        "id": "msg_" + (body["id"] if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": number_or_zero(usage.get("prompt_tokens")),
            "output_tokens": number_or_zero(usage.get("completion_tokens")),
        },
    }


def openai_json_with_model(body, model):
    return {**body, "model": model}


# ---------------------------------------------------------------- SSE helpers and stream translation


def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def iter_frames(chunks):
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def frame_with_model(frame, model):
    lines = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append("data: " + json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def openai_stream_with_model(chunks, model):
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_stream_from_openai(chunks, model):
    """A Chat Completions SSE stream as the Anthropic event sequence, in strict order:
    message_start, content_block_start/delta/stop per block, message_delta, message_stop.

    The whole upstream stream is read first so that message_start can carry the real input token
    count (Anthropic reports input_tokens in message_start; OpenAI reports it only in the trailing
    usage chunk) and message_delta the real output count — nothing is fabricated.
    """
    text_parts = []
    tool_uses = []
    input_tokens = 0
    output_tokens = 0
    finish = None

    for frame in iter_frames(chunks):
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "" or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                input_tokens = number_or_zero(chunk["usage"].get("prompt_tokens"))
                output_tokens = number_or_zero(chunk["usage"].get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            if not choices:
                continue
            choice = choices[0]
            if not is_record(choice):
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            for call in (delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []):
                if not is_record(call):
                    continue
                idx = call["index"] if is_number(call.get("index")) else len(tool_uses)
                while len(tool_uses) <= idx:
                    tool_uses.append({"id": "", "name": "", "arguments": ""})
                if isinstance(call.get("id"), str) and tool_uses[idx]["id"] == "":
                    tool_uses[idx]["id"] = call["id"]
                fn = call.get("function") if is_record(call.get("function")) else {}
                if isinstance(fn.get("name"), str) and tool_uses[idx]["name"] == "":
                    tool_uses[idx]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_uses[idx]["arguments"] += fn["arguments"]
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]

    stop_reason = stop_reason_of(finish)
    if stop_reason not in LEGAL_ANTHROPIC_STOP:
        stop_reason = "end_turn"

    events = [
        sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
    ]
    index = 0
    text = "".join(text_parts)
    if text != "":
        events.append(sse_event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}))
        events.append(sse_event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}))
        events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for i, tu in enumerate(tool_uses):
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": tu["id"] or ("toolu_%d" % i), "name": tu["name"], "input": {}},
                },
            )
        )
        if tu["arguments"]:
            events.append(
                sse_event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": tu["arguments"]}})
            )
        events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    if index == 0:
        # No text and no tool use: still close with an empty text block so the message has content.
        events.append(sse_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
        events.append(sse_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ""}}))
        events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(
        sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    events.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(events)


# ---------------------------------------------------------------- the gateway


def chat_completions_url():
    if GATEWAY_URL == "":
        return ""
    if GATEWAY_URL.endswith("/v1"):
        return GATEWAY_URL + "/chat/completions"
    return GATEWAY_URL + "/v1/chat/completions"


def dial(model, body):
    url = chat_completions_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": "Bearer " + GATEWAY_KEY}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    return response.status, response


def error_code_of(body_bytes):
    if body_bytes is None:
        return ""
    try:
        value = json.loads(body_bytes.decode("utf-8", "replace"))
    except Exception:
        return ""
    if not is_record(value) or not is_record(value.get("error")):
        return ""
    err = value["error"]
    code = err.get("code")
    if isinstance(code, str) and code != "":
        return code[:64]
    typ = err.get("type")
    if isinstance(typ, str):
        return typ[:64]
    return ""


# ---------------------------------------------------------------- trial state

BRIEF = read_brief(PROMPT_FILE)
FREE = free_models(BRIEF)
CATALOG = build_catalog(FREE)
SERVED = served_models(FREE)

# Permanently unavailable models (a refusal the trial will not recover from) and a per-model 429
# counter. Both are functions of the gateway answers seen so far, never the wall clock, so a replay
# leaves them the same.
DEAD = set()
REFUSAL_COUNT = {}
DEAD_LOCK = threading.Lock()

MANIFEST_INDEX = 0
MANIFEST_LOCK = threading.Lock()


def next_manifest_requires():
    global MANIFEST_INDEX
    with MANIFEST_LOCK:
        index = MANIFEST_INDEX
        MANIFEST_INDEX += 1
    if index < len(BRIEF["prompts"]):
        return BRIEF["prompts"][index]["requires"]
    return None


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/1"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    # -- plumbing ---------------------------------------------------------

    def _dispatch(self):
        self.headers_sent = False
        path = urllib.parse.urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self._models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self._chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self._messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self._send_error(wire, 404, "not_found", "no route %s %s" % (self.command, path))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as err:
            log("request failed: %r" % err)
            if not self.headers_sent:
                wire = "anthropic" if urllib.parse.urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self._send_error(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def _read_body(self):
        length = self.headers.get("content-length")
        if length is None:
            return None
        try:
            size = int(length)
        except ValueError:
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            return None
        return self.rfile.read(size) if size > 0 else b""

    def _read_json(self):
        body = self._read_body()
        if body is None or body == b"":
            return None
        try:
            value = json.loads(body.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _send_json(self, status, value):
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, wire, status, error_type, message):
        if wire == "anthropic":
            self._send_json(status, {"type": "error", "error": {"type": error_type, "message": message}})
        else:
            self._send_json(status, {"error": {"message": message, "type": error_type}})

    def _stream_headers(self, trail):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        if trail is not None:
            self.send_header("x-makefun-upstream-failures", json.dumps(trail, separators=(",", ":"), ensure_ascii=False))
        self.send_header("connection", "close")
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def _write_event(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    # -- routes -----------------------------------------------------------

    def _models(self):
        if self.headers.get("x-api-key") is not None:
            self._send_json(
                200,
                {
                    "data": [
                        {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
                        for m in SERVED
                    ],
                    "has_more": False,
                },
            )
        else:
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]} for m in SERVED],
                },
            )

    # -- completion serving ----------------------------------------------

    def _serve(self, client_wire, body):
        stream = body.get("stream") is True
        requires = requirements_for(body, next_manifest_requires())
        trail = []
        for model in candidate_models(requires):
            mid = model["id"]
            upstream_body = anthropic_to_openai(body, mid) if client_wire == "anthropic" else openai_to_openai(body, mid)
            try:
                status, upstream = dial(mid, upstream_body)
            except Exception:
                # An unreachable gateway is not a refusal the trail can report; keep the model
                # available and try the next one.
                continue
            if status >= 400:
                try:
                    body_bytes = upstream if isinstance(upstream, (bytes, bytearray)) else upstream.read()
                except Exception:
                    body_bytes = b""
                if not isinstance(upstream, (bytes, bytearray)):
                    try:
                        upstream.close()
                    except Exception:
                        pass
                if len(trail) < 8:
                    trail.append({"status": status, "code": error_code_of(body_bytes), "model": mid})
                with DEAD_LOCK:
                    if status in PERMANENT_REFUSAL_STATUS:
                        DEAD.add(mid)
                    elif status == 429:
                        REFUSAL_COUNT[mid] = REFUSAL_COUNT.get(mid, 0) + 1
                        if REFUSAL_COUNT[mid] >= MAX_429_BEFORE_DEAD:
                            DEAD.add(mid)
                continue
            # Success: serve the gateway's answer back in the client's wire.
            if stream:
                self._stream_headers(trail if trail else None)
                try:
                    if client_wire == "openai":
                        for frame in openai_stream_with_model(upstream_chunks(upstream), mid):
                            self._write_event(frame)
                    else:
                        self._write_event(anthropic_stream_from_openai(upstream_chunks(upstream), mid))
                finally:
                    close_upstream(upstream)
                return
            try:
                body_bytes = upstream.read()
            finally:
                close_upstream(upstream)
            try:
                parsed = json.loads(body_bytes.decode("utf-8", "replace"))
            except Exception:
                parsed = None
            if not is_record(parsed):
                # A 2xx that is not one JSON object: treat it as an unusable answer and fail over.
                continue
            if client_wire == "anthropic":
                answer = openai_to_anthropic_answer(parsed, mid)
            else:
                answer = openai_json_with_model(parsed, mid)
            self._send_json_with_trail(200, answer, trail)
            return
        # Every candidate refused or broke: give up.
        if client_wire == "anthropic":
            self._send_error("anthropic", 503, "overloaded_error", "no free model on the brief meets this request")
        else:
            self._send_error("openai", 503, "no_free_model", "no free model on the brief meets this request")

    def _send_json_with_trail(self, status, value, trail):
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        if trail:
            self.send_header("x-makefun-upstream-failures", json.dumps(trail, separators=(",", ":"), ensure_ascii=False))
        self.send_header("content-length", str(len(raw)))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def _chat(self):
        body = self._read_json()
        if body is None:
            self._send_error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self._serve("openai", body)

    def _messages(self):
        body = self._read_json()
        if body is None:
            self._send_error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self._serve("anthropic", body)


def upstream_chunks(upstream):
    reader = getattr(upstream, "read1", None) or upstream.read
    while True:
        chunk = reader(65536)
        if not chunk:
            break
        yield chunk


def close_upstream(upstream):
    try:
        upstream.close()
    except Exception:
        pass


# ---------------------------------------------------------------- register and poll


def register(port, catalog):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENV_URL + "/register", data=body, headers={"content-type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (ENV_URL, err.code, err.read().decode("utf-8", "replace")[:500])) from err
    if not (200 <= status < 300):
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (ENV_URL, status, reply_text[:500]))
    try:
        reply = json.loads(reply_text)
    except Exception:
        reply = None
    if is_record(reply) and is_number(reply.get("requests")):
        log("registered: the trial holds %s prompts" % reply["requests"])


def poll(server):
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(ENV_URL + "/work", timeout=5) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_):
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------- start


def start():
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log("listening on 0.0.0.0:%s with %d free models" % (port, len(SERVED)))
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, args=(server,), name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main():
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException as err:
        log("fatal: %r" % err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
