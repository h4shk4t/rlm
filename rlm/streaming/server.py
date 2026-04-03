import argparse
import json
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.streaming.event_bus import EventBus


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower().strip()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _default_backend() -> str:
    explicit_default = os.getenv("RLM_DEFAULT_BACKEND")
    if explicit_default:
        return explicit_default

    has_azure_endpoint = bool(
        os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPEN_AI_ENDPOINT")
    )
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_azure_endpoint and not has_openai_key:
        return "azure_openai"

    return "openai"


def _extract_iteration_tokens(iteration_entry: dict[str, Any]) -> tuple[int, int]:
    total_input = 0
    total_output = 0
    for code_block in iteration_entry.get("code_blocks", []):
        result = code_block.get("result", {})
        for call in result.get("rlm_calls", []):
            usage = call.get("usage_summary", {})
            model_summaries = usage.get("model_usage_summaries", {})
            for summary in model_summaries.values():
                total_input += _safe_int(summary.get("total_input_tokens"), 0)
                total_output += _safe_int(summary.get("total_output_tokens"), 0)
    return total_input, total_output


@dataclass
class RunSpec:
    run_id: str
    prompt: str
    root_prompt: str | None
    backend: str
    backend_kwargs: dict[str, Any]
    environment: str
    environment_kwargs: dict[str, Any]
    max_depth: int
    max_iterations: int
    max_concurrent_subcalls: int
    verbose: bool
    log_dir: str | None


class RunManager:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def start_run(self, payload: dict[str, Any]) -> str:
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Body must include a non-empty string `prompt`")

        run_id = str(uuid.uuid4())
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Optional `config` field must be a JSON object")

        backend = str(config.get("backend", _default_backend()))
        backend_kwargs = config.get("backend_kwargs", {}) or {}
        if not isinstance(backend_kwargs, dict):
            raise ValueError("`config.backend_kwargs` must be a JSON object when provided")
        if backend == "openai" and not backend_kwargs.get("model_name"):
            backend_kwargs = {
                "model_name": "",
                **backend_kwargs,
            }
        if backend == "azure_openai" and not backend_kwargs.get("model_name"):
            default_azure_model = (
                os.getenv("AZURE_OPENAI_DEPLOYMENT")
                or os.getenv("AZURE_OPEN_AI_DEPLOYMENT")
                or "gpt-4o-mini"
            )
            backend_kwargs = {
                "model_name": default_azure_model,
                **backend_kwargs,
            }

        environment_kwargs = config.get("environment_kwargs", {}) or {}
        if not isinstance(environment_kwargs, dict):
            raise ValueError("`config.environment_kwargs` must be a JSON object when provided")

        log_dir = config.get("log_dir")
        if log_dir is not None and not isinstance(log_dir, str):
            raise ValueError("`config.log_dir` must be a string path when provided")

        root_prompt = payload.get("root_prompt")
        if root_prompt is not None and not isinstance(root_prompt, str):
            raise ValueError("`root_prompt` must be a string when provided")

        spec = RunSpec(
            run_id=run_id,
            prompt=prompt,
            root_prompt=root_prompt,
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=str(config.get("environment", "local")),
            environment_kwargs=environment_kwargs,
            max_depth=_safe_int(config.get("max_depth"), 3),
            max_iterations=_safe_int(config.get("max_iterations"), 8),
            max_concurrent_subcalls=_safe_int(config.get("max_concurrent_subcalls"), 4),
            verbose=_safe_bool(config.get("verbose"), False),
            log_dir=log_dir,
        )

        thread = threading.Thread(
            target=self._run_rlm_completion,
            args=(spec,),
            daemon=True,
            name=f"rlm-run-{run_id[:8]}",
        )
        thread.start()
        return run_id

    def _emit(self, run_id: str, event_type: str, **payload: Any) -> None:
        self.event_bus.publish(
            {
                "run_id": run_id,
                "event_type": event_type,
                "timestamp": _utc_now_iso(),
                **payload,
            }
        )

    def _run_rlm_completion(self, spec: RunSpec) -> None:
        run_start = time.perf_counter()
        node_counter = 0
        parent_by_node: dict[str, str | None] = {"root": None}
        latest_node_for_depth: dict[int, str] = {0: "root"}
        pending_nodes_by_depth: dict[int, deque[str]] = defaultdict(deque)
        thread_to_node: dict[int, str] = {}  # thread_id -> node_id for concurrent subcalls
        active_node_id = "root"

        self._emit(
            spec.run_id,
            "node_start",
            node_id="root",
            parent_id=None,
            depth=0,
            model=spec.backend_kwargs.get("model_name", "unknown"),
            prompt_preview=spec.prompt[:220],
        )
        self._emit(
            spec.run_id,
            "metrics",
            active_node_id=active_node_id,
            recursion_depth=0,
            phase="run_start",
        )

        def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
            nonlocal node_counter, active_node_id
            node_counter += 1
            node_id = f"node-{node_counter}"
            parent_id = latest_node_for_depth.get(max(depth - 1, 0), "root")
            parent_by_node[node_id] = parent_id
            pending_nodes_by_depth[depth].append(node_id)
            latest_node_for_depth[depth] = node_id
            active_node_id = node_id
            # Track thread -> node mapping for concurrent subcalls
            tid = threading.current_thread().ident
            if tid is not None:
                thread_to_node[tid] = node_id

            self._emit(
                spec.run_id,
                "node_start",
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                model=model,
                prompt_preview=prompt_preview[:500],
            )
            self._emit(
                spec.run_id,
                "metrics",
                active_node_id=node_id,
                recursion_depth=depth,
                phase="subcall_start",
            )

        def on_subcall_complete(
            depth: int, model: str, duration: float, error: str | None, response_preview: str | None = None,
        ) -> None:
            nonlocal active_node_id
            node_id = (
                pending_nodes_by_depth[depth].popleft()
                if pending_nodes_by_depth[depth]
                else latest_node_for_depth.get(depth, f"depth-{depth}")
            )
            parent_id = parent_by_node.get(node_id)
            active_node_id = latest_node_for_depth.get(max(depth - 1, 0), "root")
            # Clean up thread mapping
            tid = threading.current_thread().ident
            if tid is not None:
                thread_to_node.pop(tid, None)

            self._emit(
                spec.run_id,
                "node_complete",
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                model=model,
                duration_ms=int(duration * 1000),
                error=error,
                response_preview=response_preview,
            )
            self._emit(
                spec.run_id,
                "metrics",
                active_node_id=active_node_id,
                recursion_depth=max(depth - 1, 0),
                latency_ms=int(duration * 1000),
                phase="subcall_complete",
            )
            if error:
                self._emit(
                    spec.run_id,
                    "error",
                    node_id=node_id,
                    depth=depth,
                    message=error,
                )

        def on_iteration_start(depth: int, iteration_num: int) -> None:
            # Use thread mapping for correct node attribution with concurrent subcalls
            tid = threading.current_thread().ident
            node_id = thread_to_node.get(tid) if tid is not None else None
            if node_id is None:
                node_id = latest_node_for_depth.get(depth, "root")
            self._emit(
                spec.run_id,
                "metrics",
                active_node_id=node_id,
                recursion_depth=depth,
                iteration=iteration_num,
                phase="iteration_start",
            )

        def on_iteration_complete(depth: int, iteration_num: int, duration: float, response: str, code_block_count: int, prompt_preview: str = "") -> None:
            # Use thread mapping for correct node attribution with concurrent subcalls
            tid = threading.current_thread().ident
            node_id = thread_to_node.get(tid) if tid is not None else None
            if node_id is None:
                node_id = latest_node_for_depth.get(depth, "root")
            self._emit(
                spec.run_id,
                "iteration",
                node_id=node_id,
                depth=depth,
                iteration=iteration_num,
                code_block_count=code_block_count,
                response_preview=response,
                prompt_preview=prompt_preview,
                iteration_time_ms=int(duration * 1000),
            )
            self._emit(
                spec.run_id,
                "metrics",
                active_node_id=node_id,
                recursion_depth=depth,
                iteration=iteration_num,
                latency_ms=int(duration * 1000),
                phase="iteration_complete",
            )

        def logger_event_sink(event: dict[str, Any]) -> None:
            event_type = event.get("event_type")
            if event_type == "metadata":
                self._emit(
                    spec.run_id,
                    "metrics",
                    active_node_id="root",
                    recursion_depth=0,
                    root_model=event.get("root_model"),
                    max_depth=event.get("max_depth"),
                    max_iterations=event.get("max_iterations"),
                    phase="metadata",
                )
            elif event_type == "iteration":
                # Iteration data is now emitted by on_iteration_complete callback
                # with correct node_id mapping. Extract tokens here for metrics only.
                token_in, token_out = _extract_iteration_tokens(event)
                if token_in > 0 or token_out > 0:
                    self._emit(
                        spec.run_id,
                        "metrics",
                        active_node_id="root",
                        recursion_depth=0,
                        subcall_input_tokens=token_in,
                        subcall_output_tokens=token_out,
                        phase="iteration_tokens",
                    )

        logger = RLMLogger(log_dir=spec.log_dir, event_sink=logger_event_sink)

        try:
            rlm = RLM(
                backend=spec.backend,
                backend_kwargs=spec.backend_kwargs,
                environment=spec.environment,
                environment_kwargs=spec.environment_kwargs,
                max_depth=spec.max_depth,
                max_iterations=spec.max_iterations,
                max_concurrent_subcalls=spec.max_concurrent_subcalls,
                logger=logger,
                verbose=spec.verbose,
                on_subcall_start=on_subcall_start,
                on_subcall_complete=on_subcall_complete,
                on_iteration_start=on_iteration_start,
                on_iteration_complete=on_iteration_complete,
            )

            completion = rlm.completion(spec.prompt, root_prompt=spec.root_prompt)
            total_duration_ms = int((time.perf_counter() - run_start) * 1000)
            usage = completion.usage_summary
            self._emit(
                spec.run_id,
                "metrics",
                active_node_id="root",
                recursion_depth=0,
                latency_ms=total_duration_ms,
                tokens={
                    "input": usage.total_input_tokens,
                    "output": usage.total_output_tokens,
                },
                phase="run_complete",
            )
            self._emit(
                spec.run_id,
                "node_complete",
                node_id="root",
                parent_id=None,
                depth=0,
                model=completion.root_model,
                duration_ms=total_duration_ms,
                error=None,
                response_preview=completion.response[:2000],
            )
            self._emit(
                spec.run_id,
                "run_complete",
                status="success",
                duration_ms=total_duration_ms,
                response=completion.response,
                usage_summary=completion.usage_summary.to_dict(),
            )
        except Exception as exc:
            total_duration_ms = int((time.perf_counter() - run_start) * 1000)
            error_message = str(exc)
            self._emit(
                spec.run_id,
                "error",
                node_id="root",
                depth=0,
                message=error_message,
            )
            self._emit(
                spec.run_id,
                "node_complete",
                node_id="root",
                parent_id=None,
                depth=0,
                model=spec.backend_kwargs.get("model_name", "unknown"),
                duration_ms=total_duration_ms,
                error=error_message,
            )
            self._emit(
                spec.run_id,
                "run_complete",
                status="error",
                duration_ms=total_duration_ms,
                error=error_message,
            )


class StreamingRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RLMStreamingServer/0.1"

    @property
    def service(self) -> "RLMStreamingServer":
        return self.server.service  # type: ignore[attr-defined]

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/events":
            self._handle_events(parsed.query)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/run":
            self._handle_run()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _handle_run(self) -> None:
        try:
            payload = self._read_json_body()
            run_id = self.service.run_manager.start_run(payload)
            self._write_json(
                HTTPStatus.ACCEPTED,
                {"run_id": run_id, "events_url": f"/events?run_id={run_id}"},
            )
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_events(self, query_string: str) -> None:
        query = parse_qs(query_string)
        run_id = query.get("run_id", [None])[0]
        subscription = self.service.event_bus.subscribe(run_id=run_id)

        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                try:
                    event = subscription.queue.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue

                event_type = str(event.get("event_type", "message"))
                payload = json.dumps(event, default=str)
                chunk = f"event: {event_type}\ndata: {payload}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        finally:
            self.service.event_bus.unsubscribe(subscription.token)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = _safe_int(self.headers.get("Content-Length"), 0)
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        decoded = body.decode("utf-8")
        data = json.loads(decoded)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        # Keep default HTTP handler logs quiet for local developer use.
        return


class _ThreadedServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RLMStreamingServer:
    def __init__(self):
        self.event_bus = EventBus()
        self.run_manager = RunManager(self.event_bus)


def create_streaming_server(host: str = "127.0.0.1", port: int = 8765) -> _ThreadedServer:
    service = RLMStreamingServer()
    server = _ThreadedServer((host, port), StreamingRequestHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RLM live event streaming server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = create_streaming_server(host=args.host, port=args.port)
    print(f"RLM streaming server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
