"""
DashboardServer — lightweight FastAPI server that streams live training
metrics to any browser tab via Server-Sent Events (SSE).

The server runs in a daemon background thread inside the training process.
The training loop calls ``server.emit(data)`` (a plain dict) at the end of
every update; the server fans that event out to every connected browser.

Usage (from main.py or hooks.py)
---------------------------------
    from src.dashboard.server import DashboardServer

    server = DashboardServer(host="0.0.0.0", port=6006)
    server.start()

    # later, in the training loop:
    server.emit({"type": "update", "policy_loss": 0.12, ...})

    # optionally stop cleanly:
    server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import deque
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Max history events sent to a newly connected client (catch-up)
_MAX_HISTORY: int = 5000


class DashboardServer:
    """
    Thread-safe SSE dashboard.

    All public methods (``emit``, ``start``, ``stop``) are safe to call
    from the training thread or any other thread.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 6006) -> None:
        self.host = host
        self.port = port

        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server_task = None

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def emit(self, data: Dict[str, Any]) -> None:
        """Push *data* to all connected clients.  Safe to call from any thread."""
        msg = json.dumps(data, default=lambda o: float(o) if hasattr(o, '__float__') else int(o) if hasattr(o, '__int__') else repr(o))
        with self._lock:
            self._history.append(msg)
            subs = list(self._subscribers)

        if self._loop is None or self._loop.is_closed():
            return
        for q in subs:
            asyncio.run_coroutine_threadsafe(q.put(msg), self._loop)

    def start(self) -> None:
        """Start the ASGI server in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_server,
            args=(ready,),
            daemon=True,
            name="DashboardServer",
        )
        self._thread.start()
        ready.wait(timeout=10)
        print(f"[Dashboard] Running at http://{self.host}:{self.port}")

    def stop(self) -> None:
        """Signal the background server to shut down."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_server(self, ready: threading.Event) -> None:
        """Entry point for the background thread."""
        import uvicorn  # deferred — only needed when dashboard is on

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = self._build_app()

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            loop="none",        # we manage the loop ourselves
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        async def _serve():
            ready.set()
            await server.serve()

        loop.run_until_complete(_serve())

    def _subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.add(q)
        return q

    def _unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _build_app(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles

        app = FastAPI(title="DaVinci Training Dashboard", docs_url=None, redoc_url=None)

        static_dir = os.path.join(os.path.dirname(__file__), "static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            html_path = os.path.join(static_dir, "index.html")
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

        @app.get("/history")
        async def history():
            with self._lock:
                items = list(self._history)
            return JSONResponse([json.loads(m) for m in items])

        @app.get("/stream")
        async def stream():
            q = self._subscribe()

            async def _event_generator():
                # Send history so a new tab can catch up instantly
                with self._lock:
                    hist = list(self._history)
                for msg in hist:
                    yield f"data: {msg}\n\n"
                # Then stream live events — keep loop alive on timeout (just send ping)
                try:
                    while True:
                        try:
                            msg = await asyncio.wait_for(q.get(), timeout=30)
                            yield f"data: {msg}\n\n"
                        except asyncio.TimeoutError:
                            yield "data: {\"type\": \"ping\"}\n\n"  # keepalive, don't exit
                except asyncio.CancelledError:
                    pass
                finally:
                    self._unsubscribe(q)

            return StreamingResponse(
                _event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Serve static files (JS, CSS) if the directory exists
        if os.path.isdir(static_dir):
            app.mount("/static", StaticFiles(directory=static_dir), name="static")

        return app
