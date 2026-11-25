"""Transport adapters for an independently implemented memory query benchmark.

One adapter is opened for a whole benchmark job.  Query fixtures are never
serialized to real targets: only the query, scope and requested result count
are available to the request mapper.  The mock is a harness demonstration and
deliberately uses fixture answers; its results are not retrieval measurements.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
import random
import re
import time
from typing import Any, AsyncIterator, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

from .models import QueryCase, QueryResult, TargetConfig


HARD_MAX_RESPONSE_BYTES = 2_000_000
_PLACEHOLDER = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GOLD_KEYS = {"expected_ids", "forbidden_ids"}


class AdapterError(Exception):
    """A target or adapter failure, never a successful empty result."""


class ConfigError(AdapterError):
    """Invalid or incomplete local target configuration."""


class TargetError(AdapterError):
    """The remote target or local target process did not complete the call."""


class ResponseSchemaError(AdapterError):
    """The response could not be interpreted under the configured schema."""


class ResponseTooLarge(AdapterError):
    """The response exceeded the configured or hard size limit."""


class WriteDisabledError(AdapterError):
    """Mutation was not explicitly enabled for a benchmark namespace."""


class Adapter(Protocol):
    async def query(self, case: QueryCase, top_k: int) -> QueryResult: ...

    async def upsert(self, record_id: str, text: str, scope: str) -> None: ...

    async def delete(self, record_id: str, scope: str) -> None: ...


def render_template(template: Any, values: Mapping[str, Any]) -> Any:
    """Copy a JSON template, replacing only complete scalar placeholders.

    Keys are never interpolated.  For example ``"$top_k"`` becomes an integer,
    while ``"prefix $query"`` stays literal.  Gold-label field names are
    prohibited even when supplied as static template values.
    """
    if isinstance(template, dict):
        output = {}
        for key, value in template.items():
            if not isinstance(key, str):
                raise ConfigError("request template keys must be strings")
            if key.casefold() in _GOLD_KEYS:
                raise ConfigError("gold-label fields must not be sent to a target")
            output[key] = render_template(value, values)
        return output
    if isinstance(template, list):
        return [render_template(value, values) for value in template]
    if isinstance(template, str):
        if _PLACEHOLDER.fullmatch(template):
            if template not in values:
                raise ConfigError("unknown or unavailable template placeholder")
            return values[template]
        return template
    if template is None or isinstance(template, (bool, int, float)):
        try:
            json.dumps(template, allow_nan=False)
        except (ValueError, TypeError):
            raise ConfigError("request template must contain finite JSON values") from None
        return template
    raise ConfigError("request template must contain only JSON values")


def _lookup(value: Any, path: str, label: str) -> Any:
    if path in ("", "$"):
        return value
    for part in path.split("."):
        if not part or not isinstance(value, dict) or part not in value:
            raise ResponseSchemaError(f"{label} does not resolve in the response")
        value = value[part]
    return value


def extract_result_ids(payload: Any, results_path: str, id_field: str) -> list[str]:
    """Read an array of result objects (or strings with an empty id_field)."""
    results = _lookup(payload, results_path, "results_path")
    if not isinstance(results, list):
        raise ResponseSchemaError("results_path must resolve to an array")
    ids: list[str] = []
    for item in results:
        record_id = _lookup(item, id_field, "id_field")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ResponseSchemaError("every result ID must be a nonempty string")
        ids.append(record_id)
    return ids


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ResponseSchemaError("response is not valid finite JSON") from None


def _limit_for(target: TargetConfig) -> int:
    return min(target.max_response_bytes, HARD_MAX_RESPONSE_BYTES)


def _query_arguments(target: TargetConfig, case: QueryCase, top_k: int) -> dict[str, Any]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ConfigError("top_k must be a positive integer")
    return render_template(target.query_template, {
        "$query": case.query, "$top_k": top_k, "$scope": case.scope,
    })


def _template_placeholders(template: Any) -> set[str]:
    if isinstance(template, dict):
        return set().union(*(_template_placeholders(value) for value in template.values()))
    if isinstance(template, list):
        return set().union(*(_template_placeholders(value) for value in template))
    if isinstance(template, str) and _PLACEHOLDER.fullmatch(template):
        return {template}
    return set()


def _validate_write_templates(target: TargetConfig) -> None:
    # Validate cleanup before an upsert can take effect.  Keys, substrings and
    # static values cannot stand in for the actual randomly generated sentinel.
    for label, template, required in (
        ("write", target.write_template, {"$record_id", "$scope", "$text"}),
        ("delete", target.delete_template, {"$record_id", "$scope"}),
    ):
        if not required <= _template_placeholders(template):
            raise ConfigError(f"{label} template is missing required exact placeholders")


def _write_arguments(
    target: TargetConfig, record_id: str, scope: str, text: str | None = None,
) -> dict[str, Any]:
    if not target.write_enabled:
        raise WriteDisabledError("target writes are disabled")
    if not scope.startswith("bench-"):
        raise WriteDisabledError("writes are restricted to bench- namespaces")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ConfigError("record_id must be a nonempty string")
    _validate_write_templates(target)
    values: dict[str, Any] = {"$record_id": record_id, "$scope": scope}
    template = target.delete_template
    if text is not None:
        values["$text"] = text
        template = target.write_template
    return render_template(template, values)


def _bearer_headers(target: TargetConfig) -> dict[str, str]:
    if target.token_env is None:
        return {}
    if not _ENV_NAME.fullmatch(target.token_env):
        raise ConfigError("token_env must name an environment variable")
    token = os.environ.get(target.token_env)
    if not token or not token.strip():
        raise ConfigError("the configured token environment variable is not set")
    if "\r" in token or "\n" in token:
        raise ConfigError("the configured token is not a valid header value")
    return {"Authorization": "Bearer " + token}


def _validate_http_url(url: str | None, *, base: bool = False) -> str:
    if not url:
        raise ConfigError("an HTTP(S) target URL is required")
    try:
        parsed = urlsplit(url)
        valid_port = parsed.port
        del valid_port
    except ValueError:
        raise ConfigError("invalid target URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError("an absolute HTTP(S) target URL is required")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError("URL credentials are not allowed; use token_env")
    if parsed.fragment or (base and parsed.query):
        raise ConfigError("base URLs cannot contain queries or fragments")
    return url


def _endpoint(base: str, path: str) -> str:
    if not isinstance(path, str):
        raise ConfigError("HTTP endpoint paths must be strings")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment or path.startswith("//"):
        raise ConfigError("HTTP endpoint paths must stay relative to the target")
    # Keep a configured /api prefix while accepting both 'query' and '/query'.
    return base.rstrip("/") + "/" + path.lstrip("/")


def _check_target_error(payload: Any) -> None:
    if isinstance(payload, dict):
        if payload.get("ok") is False or payload.get("success") is False or payload.get("accepted") is False:
            raise TargetError("target rejected the operation")
        if payload.get("isError") is True or payload.get("error"):
            raise TargetError("target returned an operation error")


class HTTPAdapter:
    def __init__(self, target: TargetConfig, client: httpx.AsyncClient) -> None:
        self.target = target
        self.client = client
        self.base_url = _validate_http_url(target.url, base=True)
        self.limit = _limit_for(target)
        # Validate every path before the first request can be dispatched.
        for path in (target.query_path, target.write_path, target.delete_path):
            _endpoint(self.base_url, path)

    async def _post(self, path: str, arguments: dict[str, Any], *, require_json: bool = True) -> tuple[Any, int]:
        try:
            async with self.client.stream("POST", _endpoint(self.base_url, path), json=arguments) as response:
                if not 200 <= response.status_code < 300:
                    raise TargetError(f"target returned HTTP {response.status_code}")
                content_length = response.headers.get("content-length", "")
                if content_length.isdecimal() and int(content_length) > self.limit:
                    raise ResponseTooLarge("response exceeds the byte limit")
                parts: list[bytes] = []
                size = 0
                async for part in response.aiter_bytes(chunk_size=65_536):
                    size += len(part)
                    if size > self.limit:
                        raise ResponseTooLarge("response exceeds the byte limit")
                    parts.append(part)
                body = b"".join(parts)
        except AdapterError:
            raise
        except Exception as exc:
            # Do not forward HTTP exception messages, URLs, headers or bodies:
            # these may contain queries, credentials, or private target errors.
            raise TargetError(f"HTTP transport failed ({type(exc).__name__})") from None
        if not body and not require_json:
            return None, 0
        try:
            return json.loads(body), size
        except (ValueError, UnicodeError, RecursionError):
            if not require_json:
                return None, size
            raise ResponseSchemaError("target response is not JSON") from None

    async def query(self, case: QueryCase, top_k: int) -> QueryResult:
        payload, size = await self._post(self.target.query_path, _query_arguments(self.target, case, top_k))
        _check_target_error(payload)
        ids = extract_result_ids(payload, self.target.results_path, self.target.id_field)
        return QueryResult(ids, size, {"transport": "http", "simulated": False})

    async def upsert(self, record_id: str, text: str, scope: str) -> None:
        payload, _ = await self._post(self.target.write_path, _write_arguments(self.target, record_id, scope, text), require_json=False)
        _check_target_error(payload)

    async def delete(self, record_id: str, scope: str) -> None:
        payload, _ = await self._post(self.target.delete_path, _write_arguments(self.target, record_id, scope), require_json=False)
        _check_target_error(payload)


def parse_mcp_result(
    result: Any, limit: int = HARD_MAX_RESPONSE_BYTES, *, require_json: bool = True,
) -> tuple[Any, int]:
    """Decode structured content or one JSON text block, rejecting MCP errors.

    This cap is on the decoded SDK tool result, not a pre-parse wire cap.  The
    official MCP SDK owns stream framing and JSON parsing.
    """
    if hasattr(result, "model_dump"):
        envelope = result.model_dump(mode="json", exclude_none=True)
    elif isinstance(result, dict):
        envelope = result
    else:
        raise ResponseSchemaError("MCP result has an unsupported shape")
    if not isinstance(envelope, dict):
        raise ResponseSchemaError("MCP result has an unsupported shape")
    raw = _json_bytes(envelope)
    if len(raw) > min(limit, HARD_MAX_RESPONSE_BYTES):
        raise ResponseTooLarge("MCP tool result exceeds the byte limit")
    if envelope.get("isError") is True:
        raise TargetError("MCP tool reported an error")
    if envelope.get("structuredContent") is not None:
        return envelope["structuredContent"], len(raw)
    content = envelope.get("content")
    if not isinstance(content, list):
        raise ResponseSchemaError("MCP result must provide structured content or JSON text")
    text_blocks = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(text_blocks) != 1 or not isinstance(text_blocks[0], str):
        if not require_json:
            return None, len(raw)
        raise ResponseSchemaError("MCP result must contain exactly one JSON text block")
    try:
        return json.loads(text_blocks[0]), len(raw)
    except (ValueError, UnicodeError, RecursionError):
        if not require_json:
            return None, len(raw)
        raise ResponseSchemaError("MCP text block is not JSON") from None


class MCPAdapter:
    def __init__(self, target: TargetConfig, session: Any) -> None:
        self.target = target
        self.session = session
        self.limit = _limit_for(target)

    async def _call(self, tool: str, arguments: dict[str, Any], *, require_json: bool = True) -> tuple[Any, int]:
        if not tool.strip():
            raise ConfigError("MCP tool names must be nonempty")
        try:
            result = await self.session.call_tool(tool, arguments=arguments)
        except Exception as exc:
            raise TargetError(f"MCP transport failed ({type(exc).__name__})") from None
        return parse_mcp_result(result, self.limit, require_json=require_json)

    async def query(self, case: QueryCase, top_k: int) -> QueryResult:
        payload, size = await self._call(self.target.query_tool, _query_arguments(self.target, case, top_k))
        _check_target_error(payload)
        ids = extract_result_ids(payload, self.target.results_path, self.target.id_field)
        return QueryResult(ids, size, {"transport": self.target.kind, "simulated": False})

    async def upsert(self, record_id: str, text: str, scope: str) -> None:
        payload, _ = await self._call(self.target.write_tool, _write_arguments(self.target, record_id, scope, text), require_json=False)
        _check_target_error(payload)

    async def delete(self, record_id: str, scope: str) -> None:
        payload, _ = await self._call(self.target.delete_tool, _write_arguments(self.target, record_id, scope), require_json=False)
        _check_target_error(payload)


class MockAdapter:
    """Seeded simulated results for testing the benchmark, never a real score."""
    def __init__(self, target: TargetConfig, seed: int = 42) -> None:
        self.target = target
        self.random = random.Random(seed)
        self.records: dict[tuple[str, str], dict[str, Any]] = {}

    async def query(self, case: QueryCase, top_k: int) -> QueryResult:
        _query_arguments(self.target, case, top_k)
        # Draw before yielding so the supplied seed fixes the request-order
        # sequence even when simulated response latencies complete out of order.
        delay = self.target.mock_latency_ms + self.random.uniform(0, self.target.mock_jitter_ms)
        if self.random.random() < self.target.mock_tail_probability:
            delay += self.target.mock_tail_ms
        fail = self.random.random() < self.target.mock_error_probability
        empty = self.random.random() < self.target.mock_empty_probability
        await asyncio.sleep(delay / 1000)
        if fail:
            raise TargetError("simulated target error")
        now = time.monotonic()
        # Visibility probes use an independent record store.  Their expected
        # label must never make an unwritten or deleted sentinel appear found.
        probe = case.id.startswith("bench-probe-")
        candidates = [case.id] if probe else case.expected_ids
        controlled = probe or any((case.scope, record_id) in self.records for record_id in candidates)
        if controlled:
            ids = []
            for record_id in candidates:
                record = self.records.get((case.scope, record_id))
                if record is None or now < record["available_at"]:
                    continue
                if record["deleted_at"] is not None and now >= record["deleted_at"]:
                    continue
                ids.append(record_id)
        else:
            ids = list(case.expected_ids)
        ids = [] if empty else ids[:top_k]
        size = len(_json_bytes({"results": [{"id": value} for value in ids]}))
        return QueryResult(ids, size, {"transport": "mock", "simulated": True, "fixture_answers_used": not controlled})

    async def upsert(self, record_id: str, text: str, scope: str) -> None:
        _write_arguments(self.target, record_id, scope, text)
        self.records[(scope, record_id)] = {
            "text": text,
            "available_at": time.monotonic() + self.target.mock_visibility_delay_ms / 1000,
            "deleted_at": None,
        }

    async def delete(self, record_id: str, scope: str) -> None:
        _write_arguments(self.target, record_id, scope)
        now = time.monotonic()
        record = self.records.setdefault((scope, record_id), {"text": "", "available_at": float("inf"), "deleted_at": None})
        record["deleted_at"] = now + self.target.mock_visibility_delay_ms / 1000


@asynccontextmanager
async def open_adapter(target: TargetConfig, concurrency: int = 4, seed: int = 42) -> AsyncIterator[Adapter]:
    """Open once per job, and close in the same task that entered this context.

    MCP transports own task groups: callers must not enter this context in one
    task and exit it in another.  Initialize is complete before the adapter is
    yielded, so session setup is outside measured query latency.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigError("concurrency must be a positive integer")
    render_template(target.query_template, {"$query": "", "$scope": "", "$top_k": 1})
    if target.write_enabled:
        _validate_write_templates(target)
    if target.kind == "mock":
        yield MockAdapter(target, seed)
        return
    limits = httpx.Limits(max_connections=max(4, concurrency + 2), max_keepalive_connections=max(4, concurrency + 2))
    if target.kind == "http":
        headers = _bearer_headers(target)
        async with httpx.AsyncClient(headers=headers, trust_env=False, follow_redirects=False, timeout=None, limits=limits) as client:
            yield HTTPAdapter(target, client)
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async with AsyncExitStack() as stack:
        try:
            if target.kind == "mcp_stdio":
                if not target.command or not target.command.strip():
                    raise ConfigError("mcp_stdio requires an operator-configured command")
                # The SDK uses direct exec rather than a shell.  Do not inherit
                # all process secrets or send child stderr to application logs.
                stderr = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                params = StdioServerParameters(command=target.command, args=target.args, cwd=target.cwd)
                read, write = await stack.enter_async_context(stdio_client(params, errlog=stderr))
            elif target.kind == "mcp_http":
                url = _validate_http_url(target.url)
                headers = _bearer_headers(target)

                def client_factory(headers=None, timeout=None, auth=None):
                    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False, follow_redirects=False, limits=limits)

                read, write, _get_session_id = await stack.enter_async_context(streamablehttp_client(url, headers=headers, httpx_client_factory=client_factory))
            else:
                raise ConfigError("unsupported adapter type")
            session = await stack.enter_async_context(ClientSession(read, write))
            async with asyncio.timeout(30):
                await session.initialize()
        except AdapterError:
            raise
        except Exception as exc:
            raise TargetError(f"MCP initialization failed ({type(exc).__name__})") from None
        yield MCPAdapter(target, session)
