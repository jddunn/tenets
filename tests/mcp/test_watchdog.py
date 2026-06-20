"""The stdio MCP watchdog self-terminates an abandoned server (orphaned parent, or
idle past the timeout) so reconnecting clients don't leak one process per reconnect."""

import threading
import time

from tenets.mcp.watchdog import should_terminate, start_idle_watchdog

# --- pure decision -----------------------------------------------------------


def test_orphan_terminates_even_when_just_active():
    stop, reason = should_terminate(idle_seconds=0.0, idle_timeout=600, ppid=1)
    assert stop and "orphan" in reason


def test_idle_terminates_past_timeout():
    stop, reason = should_terminate(idle_seconds=601, idle_timeout=600, ppid=500)
    assert stop and "idle" in reason


def test_active_server_does_not_terminate():
    stop, _ = should_terminate(idle_seconds=10, idle_timeout=600, ppid=500)
    assert not stop


def test_idle_check_disabled_when_timeout_not_positive():
    # idle disabled -> a long-idle but parented server survives ...
    stop, _ = should_terminate(idle_seconds=99999, idle_timeout=0, ppid=500)
    assert not stop
    # ... but orphan detection is always on even with idle disabled.
    stop, reason = should_terminate(idle_seconds=0, idle_timeout=0, ppid=1)
    assert stop and "orphan" in reason


# --- thread wiring (injected clocks + fake terminate, no real signal) --------


def test_watchdog_thread_fires_on_idle():
    fired = threading.Event()
    captured = {}

    def fake_terminate(reason, log):
        captured["reason"] = reason
        fired.set()

    start_idle_watchdog(
        get_last_activity=lambda: 0.0,
        idle_timeout=5.0,
        poll_interval=0.0,
        monotonic=lambda: 100.0,  # idle = 100 - 0 = 100 > 5
        getppid=lambda: 500,
        sleep=lambda s: time.sleep(0.001),
        terminate=fake_terminate,
    )
    assert fired.wait(2.0)
    assert "idle" in captured["reason"]


def test_watchdog_thread_fires_on_orphan_even_with_idle_disabled():
    fired = threading.Event()

    def fake_terminate(reason, log):
        fired.set()

    start_idle_watchdog(
        get_last_activity=lambda: 99.0,
        idle_timeout=0,  # idle disabled
        poll_interval=0.0,
        monotonic=lambda: 100.0,
        getppid=lambda: 1,  # orphaned
        sleep=lambda s: time.sleep(0.001),
        terminate=fake_terminate,
    )
    assert fired.wait(2.0)
