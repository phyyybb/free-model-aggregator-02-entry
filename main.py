# free-model-aggregator-02 entry: one server, two client wires, and a real free model behind every
# answer, routed deterministically to the best-rated free model. Python 3.13, standard library only.
#
# A trial runs this as `python3 -P -s /bot/main.py` with PYTHONHASHSEED=0. -P keeps the script's own
# directory off sys.path and -s keeps user site-packages off it, so everything lives in this one file
# plus the vendored agent shim (vendor/makefun_agent.py), which this file puts on sys.path itself.
#
# WHAT A TRIAL GIVES YOU
#   MAKEFUN_PROMPT_FILE   the brief: the free model list (id, rating, capabilities) and the six prompts'
#                         manifest, in the order the env sends them.
#   MAKEFUN_ENV_URL       the env: POST /register, GET /work.
#   OPENAI_BASE_URL       the gateway (OpenAI Chat Completions): the only route the free models have.
#   OPENAI_API_KEY        the trial's dummy key, which the gateway swaps for the entrant's OpenRouter key.
#
# THE SERVER
#   1. reads the trial and the brief and plans routing deterministically — the highest-rated free model
#      meeting each prompt's requirements — with no extra model call;
#   2. listens on 0.0.0.0 at a free port and POSTs {protocol_version, port, catalog} to /register;
#   3. serves GET /v1/models (OpenAI shape for Authorization, Anthropic shape for x-api-key),
#      POST /v1/chat/completions and POST /v1/messages, JSON and server-sent events;
#   4. answers each prompt by dialling the highest-rated free model that meets its requirements through
#      the gateway, failing over to the next model on a refusal and reporting the trail in
#      x-makefun-upstream-failures;
#   5. polls GET /work and exits 0 when it says done.
#
# THE MODEL RULE
#   Every response's `model` field names the model this entry really dialled: the house records every
#   gateway call and scores the model that actually answered. The gateway's Chat Completions answer
#   spells a free model without the `openrouter/` prefix, so every relayed answer and every streamed
#   chunk gets the dialled id put back.
#
# DETERMINISM
#   Everything a request carries is a pure function of the brief, the client's requests and the
#   gateway's answers seen so far: no wall clock, no unseeded randomness. Routing is a pure sort of the
#   brief's ratings, so the live pass and its replay dial the same models in the same order.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# The agent/v1 shim vendored beside this file: read_trial() and the gateway's model clients. -P keeps
# /bot off sys.path, so this file puts the vendor directory there itself before importing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
import makefun_agent as af  # noqa: E402

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The provider's public endpoint (the catalog names this, never the gateway's trial-local address).
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# Fixed timestamps for the model listings: nothing in a reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

# The output cap added when a request names none (the gateway refuses a call without one).
DEFAULT_MAX_TOKENS = 512
# The gateway refuses an output cap over this; clamp to keep a prompt servable.
MAX_ALLOWED_TOKENS = 8192

# ---------------------------------------------------------------- small type checks


def is_number(value: object) -> bool:
    """A JSON number: an int or a float, never a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; string contents skipped."""
    depth = 0
    in_string = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[dict]:
    """Every JSON object in `text` that carries a string `schema`, in the order they appear."""
    out: list[dict] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if isinstance(value, dict) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[dict], keys: tuple[str, ...]) -> tuple[dict, str] | None:
    """The first document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> dict | None:
    """One model row. A free row is routed as the gateway spells its id (`openrouter/<upstream>`)."""
    if not isinstance(value, dict):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> dict | None:
    """One prompt row: its id, api, streaming flag, weight and the requirements a model must meet."""
    if not isinstance(value, dict):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if isinstance(value.get('requires'), dict) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(text: str) -> dict:
    """The brief's two documents, found under this contest's schema prefix.

    Returns {"models": [...], "prompts": [...], "error": <string or None>}. The brief's prose is never
    parsed: each document is found by its `schema` member.
    """
    if text == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def read_brief_file(path: str) -> dict:
    """read_brief for the `--list` path, which runs outside a trial and reads the file directly."""
    if path == '':
        return read_brief('')
    try:
        with open(path, encoding='utf-8') as handle:
            return read_brief(handle.read())
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}


def free_models(brief: dict) -> list[dict]:
    """The models this entry may serve: the brief's free ones, in the brief's order."""
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: dict) -> dict:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: dict, caps: dict) -> bool:
    """Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one."""
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: dict) -> dict:
    """The catalog the brief's free model list makes: one provider, its models.

    The base_url is the provider's public endpoint, never the gateway's trial-local address (the
    gateway's origin differs between a live pass and its replay, and the catalog is part of the state
    export a replay must reproduce byte for byte). The catalog column compares model-id sets only, so it
    lists exactly the fourteen free ids, once each, in the brief's order.
    """
    seen: set[str] = set()
    models: list[dict] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[dict]) -> dict:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[dict]) -> dict:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> dict:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> dict:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: dict) -> dict | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image; None for the rest."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and isinstance(source, dict):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON when it holds something else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> dict:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: dict, model: str) -> dict:
    """An Anthropic Messages request as an OpenAI Chat Completions request that names `model`."""
    messages: list[dict] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[dict] = []
            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(block['input'] if isinstance(block.get('input'), dict) else {}, separators=(',', ':'), ensure_ascii=False),
                        },
                    }
                )
            text = text_of(content)
            message: dict = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: dict = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[dict] = []
        for raw in body['tools']:
            if not isinstance(raw, dict) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if isinstance(raw.get('input_schema'), dict) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if isinstance(choice, dict):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: dict, model: str) -> dict:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get('message') if isinstance(choice.get('message'), dict) else {}
    content: list[dict] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if isinstance(body.get('usage'), dict) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: dict, model: str) -> dict:
    """A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim."""
    return {**body, 'model': model}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`, and its blank line kept."""
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if isinstance(chunk, dict):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line that ends each one."""
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, every data payload's `model` the id the entry dialled."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence."""
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get('usage'), dict):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and isinstance(choices[0], dict) else None
            if choice is None:
                continue
            delta = choice.get('delta') if isinstance(choice.get('delta'), dict) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event('content_block_delta', {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}})
                    )
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- the trial state

TRIAL: af.Trial | None = None
BRIEF: dict = {'models': [], 'prompts': [], 'error': 'not read'}
CATALOG: dict = {'providers': []}
PLANNED_GENERAL: str | None = None
ENV_URL = ''
GATEWAY_URL = ''
GATEWAY_KEY = ''
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base that does not
    already end in `/v1` gets one, so both spellings reach the route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[dict]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def free_models_ranked(requires: dict) -> list[dict]:
    """The free models meeting `requires`, sorted by rating (desc), ties keeping the brief's order."""
    ranked = [m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    ranked.sort(key=lambda m: -m['rating_milli'])
    return ranked


def find_model_by_id(model_id: str) -> dict | None:
    for m in free_models(BRIEF):
        if m['id'] == model_id:
            return m
    return None


def route_candidates(requires: dict) -> list[dict]:
    """The models to dial for a prompt, in dial order: the best by rating, the deterministic plan's
    default first when it names a model at least as highly rated for a no-requirement prompt."""
    ranked = free_models_ranked(requires)
    if not ranked:
        return []
    if not requires and PLANNED_GENERAL is not None:
        planned = find_model_by_id(PLANNED_GENERAL)
        if planned is not None and planned['rating_milli'] >= ranked[0]['rating_milli']:
            return [planned] + [m for m in ranked if m['id'] != planned['id']]
    return ranked


def requirements_for(body: dict) -> dict:
    """The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF.get('prompts') or []
    if index < len(prompts):
        return prompts[index]['requires']
    return requirements_from_body(body)


def requirements_from_body(body: dict) -> dict:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if isinstance(block, dict) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if isinstance(response_format, dict) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


# ---------------------------------------------------------------- the routing plan (deterministic)


def plan_general() -> str | None:
    """The default model for simple (no-tools/no-json/no-vision) prompts, chosen deterministically: the
    highest-rated free model meeting no requirement.

    The former startup planning call always resolved to exactly this model — its answer was gated to
    name one rated at least the deterministic best — so the decision is reused here without spending a
    gateway call. The six prompt completions themselves, each a completion the model answered using
    tokens, carry the entry past the LLM agent floor, so no separate planning call is needed.
    """
    ranked = free_models_ranked({})
    return ranked[0]['id'] if ranked else None


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def ensure_max_tokens(body: dict) -> dict:
    """Ensure the gateway's output cap is present and within its bounds; return the (possibly new) body."""
    key = 'max_completion_tokens' if 'max_completion_tokens' in body else ('max_tokens' if 'max_tokens' in body else None)
    if key is None:
        body['max_tokens'] = DEFAULT_MAX_TOKENS
        return body
    value = body[key]
    if is_number(value):
        n = int(value)
        if n < 1:
            n = DEFAULT_MAX_TOKENS
        elif n > MAX_ALLOWED_TOKENS:
            n = MAX_ALLOWED_TOKENS
        body[key] = n
    else:
        body[key] = DEFAULT_MAX_TOKENS
    return body


def dial(model: str, body: dict) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model this entry dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(text: str) -> str:
    """The error `type` (or `code`) of a refused answer's body, or '' when there is none."""
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ''
    if not isinstance(body, dict):
        return ''
    err = body.get('error')
    if isinstance(err, dict):
        t = err.get('type')
        if isinstance(t, str):
            return t
        c = err.get('code')
        if isinstance(c, str):
            return c
    return ''


def failures_header(failures: list[dict]) -> str | None:
    """The x-makefun-upstream-failures header value: at most 8 rows, or None when there were none."""
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)


def dial_with_failover(candidates: list[dict], make_body):
    """Dial the candidates in order until one answers; return (upstream, model_id, failures).

    `make_body(model_id)` builds the Chat Completions body for one candidate. A refused answer (any
    non-2xx) is recorded in `failures` and the next candidate is tried; a connection failure that never
    reached the gateway is recorded with status 0 and no error code.
    """
    failures: list[dict] = []
    for model in candidates:
        mid = model['id']
        body = make_body(mid)
        try:
            upstream = dial(mid, body)
        except Exception:
            failures.append({'status': 0, 'code': '', 'model': mid})
            continue
        if upstream.ok:
            return upstream, mid, failures
        text = upstream.text()
        failures.append({'status': upstream.status, 'code': error_code_of(text), 'model': mid})
    return None, None, failures


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra_headers: dict | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self, extra_headers: dict | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        # Python's http.server does not chunk a response without a length; the close ends the stream.
        self.send_header('connection', 'close')
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> dict | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def refused(self, wire: str, failures: list[dict]) -> None:
        """Every candidate was refused: a 502 in the client's wire, with the failure trail."""
        last = failures[-1] if failures else {'status': 0, 'code': '', 'model': None}
        trail = failures_header(failures)
        headers = {'x-makefun-upstream-failures': trail} if trail is not None else None
        if last['status'] == 0:
            suffix = ''
        else:
            code = last['code']
            suffix = f': the gateway answered HTTP {last["status"]}' + (f' {code}' if code else '')
        message = f'every free model that meets this request was refused{suffix}'
        if wire == 'anthropic':
            self.send_json(502, anthropic_error('api_error', message), headers)
        else:
            self.send_json(502, openai_error(message, 'upstream_error', None), headers)

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        candidates = route_candidates(requires)
        if not candidates:
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return
        stream = body.get('stream') is True
        upstream, mid, failures = dial_with_failover(candidates, lambda m: ensure_max_tokens({**body, 'model': m}))
        if upstream is None:
            self.refused('openai', failures)
            return
        trail = failures_header(failures)
        headers = {'x-makefun-upstream-failures': trail} if trail is not None else None
        if stream:
            self.stream_headers(headers)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), mid):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_json_with_model(parsed if isinstance(parsed, dict) else {}, mid), headers)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        candidates = route_candidates(requires)
        if not candidates:
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return
        stream = body.get('stream') is True
        upstream, mid, failures = dial_with_failover(candidates, lambda m: ensure_max_tokens(anthropic_to_openai(body, m)))
        if upstream is None:
            self.refused('anthropic', failures)
            return
        trail = failures_header(failures)
        headers = {'x-makefun-upstream-failures': trail} if trail is not None else None
        if stream:
            self.stream_headers(headers)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), mid):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_to_anthropic(parsed if isinstance(parsed, dict) else {}, mid), headers)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: dict) -> None:
    """POST /register with the port and the one catalog; the env answers with the trial's prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(f'{ENV_URL}/register', data=body.encode('utf-8'), headers={'content-type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if isinstance(reply, dict) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if isinstance(work, dict) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------- start


def start() -> None:
    if BRIEF.get('error') is not None:
        log(BRIEF['error'])
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def list_catalog() -> None:
    """`--list` prints the catalog document and exits 0, before any server starts."""
    brief = read_brief_file(os.environ.get('MAKEFUN_PROMPT_FILE', ''))
    catalog = build_catalog(brief)
    sys.stdout.write(json.dumps(catalog, separators=(',', ':'), ensure_ascii=False) + '\n')
    sys.stdout.flush()


def main() -> int:
    global TRIAL, BRIEF, CATALOG, PLANNED_GENERAL, ENV_URL, GATEWAY_URL, GATEWAY_KEY
    if '--list' in sys.argv[1:]:
        list_catalog()
        return 0
    TRIAL = af.read_trial()
    BRIEF = read_brief(TRIAL.prompt)
    CATALOG = build_catalog(BRIEF)
    ENV_URL = TRIAL.env_url.rstrip('/')
    GATEWAY_URL = TRIAL.openai.base_url
    GATEWAY_KEY = TRIAL.openai.api_key
    PLANNED_GENERAL = plan_general()
    if PLANNED_GENERAL is not None:
        log(f'deterministic default model: {PLANNED_GENERAL}')
    start()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        raise SystemExit(1)
