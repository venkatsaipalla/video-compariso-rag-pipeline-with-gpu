"""Centralised logger.

Every log line is automatically prefixed with the active user_id and
session_id, captured from contextvars so they propagate through asyncio
tasks (FastAPI handlers, retrieval calls, etc.). For sync work dispatched
to other threads (e.g. ThreadPoolExecutor), re-bind via bind_context()
at the worker entry point.

Timestamp format: `Mon Nov 11 14:23:45.123` (weekday + short month + day +
HH:MM:SS.milliseconds).

Usage:
    from app.services.logger import bind_context, get_logger
    log = get_logger("ingest")          # logger name will be "app.ingest"
    bind_context(user_id="u1", session_id="s1")
    log.info("hello %s", "world")
"""
import logging
import time
from contextvars import ContextVar

_user_id_var: ContextVar[str] = ContextVar("app_user_id", default="")
_session_id_var: ContextVar[str] = ContextVar("app_session_id", default="")


def bind_context(user_id: str = "", session_id: str = "") -> None:
    """Bind user_id / session_id into the current async context. Both are
    optional; pass only the one(s) you have. Safe to call multiple times."""
    if user_id:
        _user_id_var.set(user_id)
    if session_id:
        _session_id_var.set(session_id)


def clear_context() -> None:
    _user_id_var.set("")
    _session_id_var.set("")


class _ContextFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:  # noqa: ARG002
        ct = self.converter(record.created)
        base = time.strftime("%a %b %d %H:%M:%S", ct)
        return f"{base}.{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        user = _user_id_var.get() or "-"
        sid = _session_id_var.get() or "-"
        sid_short = sid if len(sid) <= 12 else sid[:8] + "..."
        record.ctx = f"[user={user} session={sid_short}]"
        return super().format(record)


_FORMAT = "%(asctime)s %(ctx)s [%(name)s] %(levelname)s: %(message)s"


def _configure() -> None:
    base = logging.getLogger("app")
    if getattr(base, "_app_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_ContextFormatter(_FORMAT))
    base.addHandler(handler)
    base.setLevel(logging.INFO)
    base.propagate = False
    base._app_configured = True  # type: ignore[attr-defined]


_configure()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'app.' namespace with the centralised
    context-aware formatter attached."""
    if not name.startswith("app."):
        name = f"app.{name}"
    return logging.getLogger(name)
