"""Self-termination watchdog for the stdio MCP server.

A stdio MCP server already exits when its stdin hits EOF or it receives SIGTERM.
But an MCP *client* that reconnects may spawn a fresh server and abandon the old
one with the pipe still held open (so no EOF arrives) and without sending SIGTERM.
The abandoned server then blocks forever on a read that never completes — it leaks,
accumulating one stray process per reconnect.

This watchdog is the standard safety net for that case: a daemon thread that
terminates the process when it is clearly abandoned —

  * orphaned — the parent process exited (the server is reparented to PID 1), or
  * idle — no MCP request (ping / list / call) arrived for longer than the timeout.

The client transparently respawns the server on next use, so termination is safe.
Orphan detection is always on (a dead parent is unambiguous); the idle check is
opt-out via ``idle_timeout <= 0``.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable, Optional, Tuple

# On Unix an orphaned process is reparented to init/launchd (PID 1).
ORPHAN_PPID = 1


def should_terminate(idle_seconds: float, idle_timeout: float, ppid: int) -> Tuple[bool, str]:
    """Pure decision: should the server self-terminate now?

    Orphan detection (``ppid == 1``) is always active. The idle check fires only
    when ``idle_timeout > 0`` and the server has been idle longer than it.
    """
    if ppid == ORPHAN_PPID:
        return True, "orphaned (parent process exited)"
    if idle_timeout > 0 and idle_seconds > idle_timeout:
        return True, f"idle {idle_seconds:.0f}s > {idle_timeout:.0f}s timeout"
    return False, ""


def _default_terminate(reason: str, log: Optional[Callable[[str], None]]) -> None:
    if log:
        log(f"tenets-mcp watchdog: terminating — {reason}")
    # SIGTERM (not os._exit): the server handles it gracefully and the OS delivers
    # it to the main thread, unwinding the run loop and flushing the disk index.
    os.kill(os.getpid(), signal.SIGTERM)


def start_idle_watchdog(
    get_last_activity: Callable[[], float],
    idle_timeout: float,
    *,
    poll_interval: float = 20.0,
    log: Optional[Callable[[str], None]] = None,
    monotonic: Callable[[], float] = time.monotonic,
    getppid: Callable[[], int] = os.getppid,
    sleep: Callable[[float], None] = time.sleep,
    terminate: Optional[Callable[[str, Optional[Callable[[str], None]]], None]] = None,
) -> threading.Thread:
    """Start a daemon thread that self-terminates the process when abandoned.

    ``get_last_activity`` returns the ``time.monotonic()`` timestamp of the most
    recent MCP request. The clock / ppid / sleep / terminate seams are injectable
    so the loop is testable without real threads-of-control or signals.
    """
    terminate = terminate or _default_terminate

    def _loop() -> None:
        while True:
            sleep(poll_interval)
            idle = monotonic() - get_last_activity()
            stop, reason = should_terminate(idle, idle_timeout, getppid())
            if stop:
                terminate(reason, log)
                return

    thread = threading.Thread(target=_loop, name="tenets-mcp-watchdog", daemon=True)
    thread.start()
    return thread
