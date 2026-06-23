"""
ComfyUI communication layer for the editable-AI-image backend.

Phase 1 — pure transport. This module knows ONLY how to talk to a locally
running ComfyUI server over its HTTP API. It contains NO editing logic, no
workflow graphs, and no knowledge of the editor's objects/layers. It is fully
isolated from the existing inpaint service (app.py, lama_engine.py, …) so that
wiring AI features in later requires no changes to working code.

ComfyUI HTTP API used (default http://127.0.0.1:8188):

  GET  /system_stats          — server / version / device info  (health probe)
  GET  /queue                 — running + pending queue
  POST /prompt                — queue a workflow graph -> { prompt_id, ... }
  GET  /history/{prompt_id}   — execution result for a queued prompt
  GET  /view                  — fetch an output image (raw bytes)
  POST /upload/image          — upload an input image
  POST /interrupt             — interrupt the running prompt
  GET  /object_info           — node schema catalogue

Offline handling: every call goes through `_request`, which converts connection
and timeout failures into `ComfyUIUnavailable`, so callers get one predictable
error type when ComfyUI is not running. `is_available()` and `status()` never
raise — they are safe to call from a health endpoint.

Config (env vars):
  COMFYUI_URL      — base URL of the ComfyUI server (default http://127.0.0.1:8188)
  COMFYUI_TIMEOUT  — per-request timeout in seconds (default 5.0)
"""

import os
import time
import uuid

import httpx

DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
DEFAULT_TIMEOUT = float(os.environ.get("COMFYUI_TIMEOUT", "5.0"))


class ComfyUIError(Exception):
    """Base class for any ComfyUI communication failure."""


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI could not be reached (offline, wrong URL, or timed out)."""


class ComfyUIRequestError(ComfyUIError):
    """ComfyUI was reached but returned an error status."""

    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ComfyUIClient:
    """
    Thin synchronous client for a local ComfyUI server.

    Synchronous on purpose: it mirrors the existing backend's blocking style and
    lets FastAPI run calls in its threadpool without any async plumbing. A single
    httpx.Client is created lazily so importing this module opens no sockets.
    """

    def __init__(self, base_url=None, timeout=None, client_id=None):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT
        # Stable id so progress/queue can be attributed to this client later.
        self.client_id = client_id or str(uuid.uuid4())
        self._http = None

    # -- transport -----------------------------------------------------------

    @property
    def http(self):
        if self._http is None:
            self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._http

    def close(self):
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _request(self, method, path, **kwargs):
        """Issue one HTTP request, normalising every failure mode.

        Connection/timeout problems -> ComfyUIUnavailable.
        Reached-but-errored responses -> ComfyUIRequestError (with body payload).
        """
        try:
            resp = self.http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.PoolTimeout, httpx.TransportError) as exc:
            raise ComfyUIUnavailable(
                f"ComfyUI not reachable at {self.base_url} ({exc.__class__.__name__})"
            ) from exc

        if resp.status_code >= 400:
            payload = None
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001 - body may not be JSON
                payload = resp.text
            raise ComfyUIRequestError(
                f"ComfyUI {method} {path} -> HTTP {resp.status_code}",
                status_code=resp.status_code,
                payload=payload,
            )
        return resp

    # -- health --------------------------------------------------------------

    def system_stats(self):
        """Raw /system_stats payload. Raises ComfyUIUnavailable if offline."""
        return self._request("GET", "/system_stats").json()

    def is_available(self):
        """True if ComfyUI answers the health probe. Never raises."""
        try:
            self.system_stats()
            return True
        except ComfyUIError:
            return False

    def wait_until_ready(self, timeout=240.0, interval=2.0, on_attempt=None):
        """Block until ComfyUI answers its health probe, or raise on timeout.

        Resilient startup: callers (orchestrators, warm-up code) use this so they
        never queue generation against a server that is still loading models. It
        polls `is_available()` every `interval` seconds up to `timeout`, and never
        raises anything except ComfyUIUnavailable (on timeout). `on_attempt`, if
        given, is called with (elapsed_seconds) after each failed probe — handy
        for logging a "still waiting…" diagnostic.
        """
        deadline = time.monotonic() + timeout
        start = time.monotonic()
        while True:
            if self.is_available():
                return True
            if time.monotonic() >= deadline:
                raise ComfyUIUnavailable(
                    f"ComfyUI did not become ready at {self.base_url} within {timeout:.0f}s"
                )
            if on_attempt is not None:
                try:
                    on_attempt(time.monotonic() - start)
                except Exception:  # noqa: BLE001 - diagnostics must never break the wait
                    pass
            time.sleep(interval)

    def status(self):
        """Friendly health summary for a status endpoint. Never raises.

        Returns {online: bool, url, ...} — when online, includes ComfyUI
        version, primary device and queue depth; when offline, an `error`.
        """
        summary = {"online": False, "url": self.base_url, "client_id": self.client_id}
        try:
            stats = self.system_stats()
        except ComfyUIError as exc:
            summary["error"] = str(exc)
            return summary

        system = stats.get("system", {}) or {}
        devices = stats.get("devices", []) or []
        summary["online"] = True
        summary["comfyui_version"] = system.get("comfyui_version")
        summary["python_version"] = system.get("python_version")
        summary["pytorch_version"] = system.get("pytorch_version")
        if devices:
            d = devices[0]
            summary["device"] = {
                "name": d.get("name"),
                "type": d.get("type"),
                "vram_total": d.get("vram_total"),
                "vram_free": d.get("vram_free"),
            }
        try:
            q = self.get_queue()
            summary["queue"] = {
                "running": len(q.get("queue_running", [])),
                "pending": len(q.get("queue_pending", [])),
            }
        except ComfyUIError:
            pass  # health is still "online"; queue is best-effort
        return summary

    # -- queue / execution ---------------------------------------------------

    def queue_prompt(self, graph, client_id=None, prompt_id=None, extra_data=None):
        """Queue a ComfyUI workflow `graph` (an API-format prompt dict).

        Phase 1 builds no graphs itself — callers pass their own. Returns the
        server response { prompt_id, number, node_errors }. Raises
        ComfyUIRequestError (HTTP 400) if the graph fails validation.
        """
        body = {"prompt": graph, "client_id": client_id or self.client_id}
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if extra_data is not None:
            body["extra_data"] = extra_data
        return self._request("POST", "/prompt", json=body).json()

    def get_history(self, prompt_id=None):
        """History for one prompt id, or the whole history if id is None."""
        path = "/history" if prompt_id is None else f"/history/{prompt_id}"
        return self._request("GET", path).json()

    def get_queue(self):
        """Current running + pending queue."""
        return self._request("GET", "/queue").json()

    def interrupt(self):
        """Interrupt the currently executing prompt."""
        self._request("POST", "/interrupt")
        return True

    def await_result(self, prompt_id, timeout=120.0, poll_interval=0.5):
        """Poll /history until `prompt_id` completes; return its history entry.

        Raises TimeoutError if it doesn't finish in `timeout` seconds, and
        ComfyUIUnavailable if ComfyUI drops mid-poll.
        """
        deadline = time.monotonic() + timeout
        while True:
            history = self.get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {}) or {}
                # `completed` is set once execution ends (success or error).
                if status.get("completed") is True or "outputs" in entry:
                    return entry
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ComfyUI prompt {prompt_id} did not complete within {timeout}s"
                )
            time.sleep(poll_interval)

    # -- images --------------------------------------------------------------

    def get_image(self, filename, subfolder="", image_type="output"):
        """Fetch an image ComfyUI produced or stored. Returns raw bytes."""
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        return self._request("GET", "/view", params=params).content

    def upload_image(self, image_bytes, filename, image_type="input",
                     subfolder="", overwrite=True):
        """Upload an input image to ComfyUI. Returns { name, subfolder, type }."""
        files = {"image": (filename, image_bytes, "image/png")}
        data = {"type": image_type, "subfolder": subfolder,
                "overwrite": "true" if overwrite else "false"}
        return self._request("POST", "/upload/image", files=files, data=data).json()

    # -- catalogue -----------------------------------------------------------

    def get_object_info(self, node_class=None):
        """Node schema catalogue (all nodes, or one class)."""
        path = "/object_info" if node_class is None else f"/object_info/{node_class}"
        return self._request("GET", path).json()
