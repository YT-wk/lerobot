"""Structured, line-buffered diagnostics for real-robot HIL control."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from lerobot.types import EnvTransition, TransitionKey

logger = logging.getLogger(__name__)

HIL_DIAGNOSTICS_KEY = "hil_diagnostics"


def update_hil_diagnostics(transition: EnvTransition, stage: str, payload: dict[str, Any]) -> None:
    """Attach one stage's diagnostic payload to the current transition."""
    complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
    diagnostics = dict(complementary_data.get(HIL_DIAGNOSTICS_KEY, {}))
    diagnostics[stage] = payload
    complementary_data[HIL_DIAGNOSTICS_KEY] = diagnostics
    transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


class HILDiagnosticsLogger:
    """Write timestamped HIL diagnostics to a JSON Lines file immediately."""

    def __init__(self, log_dir: str | Path, *, console_summary: bool = True) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone()
        filename_timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.log_dir / f"hil_diagnostics_{filename_timestamp}.jsonl"
        self.console_summary = console_summary
        self._lock = threading.Lock()
        self._closed = False
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self.record("logger_started", {"log_path": str(self.path)})
        print(f"[HIL-DIAG] Writing real-time diagnostics to {self.path}", flush=True)

    def record(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        console_summary: str | None = None,
    ) -> None:
        if self._closed:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        record = {
            "timestamp": timestamp,
            "unix_time_s": time.time(),
            "monotonic_time_s": time.perf_counter(),
            "event": event,
            **(payload or {}),
        }
        line = json.dumps(_json_safe(record), ensure_ascii=True, separators=(",", ":"))
        try:
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        except Exception:
            logger.exception("Failed to write HIL diagnostics to %s", self.path)
            return

        if self.console_summary and console_summary:
            print(f"[HIL-DIAG {timestamp}] {console_summary}", flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self.record("logger_stopped")
        with self._lock:
            self._file.close()
            self._closed = True
