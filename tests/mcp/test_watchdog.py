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
        get_state=lambda: (0.0, 0),  # (last_activity, inflight)
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
        get_state=lambda: (99.0, 0),
        idle_timeout=0,  # idle disabled
        poll_interval=0.0,
        monotonic=lambda: 100.0,
        getppid=lambda: 1,  # orphaned
        sleep=lambda s: time.sleep(0.001),
        terminate=fake_terminate,
    )
    assert fired.wait(2.0)


def test_inflight_request_is_never_idle():
    # idle far past the timeout, but a request is in flight -> not reaped.
    stop, _ = should_terminate(idle_seconds=99999, idle_timeout=5, ppid=500, inflight=1)
    assert not stop


def test_orphan_terminates_even_with_inflight():
    # a dead parent means the result has nowhere to go -> reap regardless of inflight.
    stop, reason = should_terminate(idle_seconds=0, idle_timeout=600, ppid=1, inflight=3)
    assert stop and "orphan" in reason


def test_watchdog_waits_for_inflight_to_drain():
    # "idle" past the timeout, but a call is in flight -> not reaped until it drains.
    fired = threading.Event()
    inflight = {"n": 1}

    def fake_terminate(reason, log):
        fired.set()

    start_idle_watchdog(
        get_state=lambda: (0.0, inflight["n"]),
        idle_timeout=5.0,
        poll_interval=0.0,
        monotonic=lambda: 100.0,  # idle = 100 > 5
        getppid=lambda: 500,
        sleep=lambda s: time.sleep(0.005),
        terminate=fake_terminate,
    )
    assert not fired.wait(0.2)  # busy -> survives despite looking idle
    inflight["n"] = 0  # request completes
    assert fired.wait(2.0)  # now reaped


def test_not_reaped_immediately_when_long_request_completes():
    # A request that ran far longer than the timeout just finished: the atomic
    # snapshot reports inflight=0 paired with a FRESH last_activity (set at
    # completion), so idle is ~0 and the server is NOT reaped at completion.
    now = 1000.0
    fired = threading.Event()
    start_idle_watchdog(
        get_state=lambda: (now, 0),  # fresh completion timestamp, drained
        idle_timeout=5.0,
        poll_interval=0.0,
        monotonic=lambda: now + 1.0,  # idle = 1s < 5s
        getppid=lambda: 500,
        sleep=lambda s: time.sleep(0.005),
        terminate=lambda r, l: fired.set(),
    )
    assert not fired.wait(0.2)
