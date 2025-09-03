"""Multiprocessing utilities for intelligent worker allocation.

This module provides utilities for determining optimal worker counts
based on system resources while being respectful of user's machine.
"""

import os
from typing import Optional


def get_optimal_workers(
    config_workers: Optional[int] = None,
    min_workers: int = 1,
    max_workers: Optional[int] = None,
    reserve_cores: int = 2,
    workload_type: str = "balanced",
) -> int:
    """Determine optimal number of workers for multiprocessing.

    This function intelligently determines worker count by:
    - Auto-detecting CPU cores
    - Reserving cores for system/user tasks
    - Respecting configured limits
    - Adjusting based on workload type

    Args:
        config_workers: User-configured worker count (None for auto)
        min_workers: Minimum workers to use
        max_workers: Maximum workers allowed (None for no limit)
        reserve_cores: Number of cores to reserve for system (default: 2)
        workload_type: Type of workload - "light", "balanced", "heavy"

    Returns:
        Optimal number of workers to use
    """
    # Get total CPU cores
    cpu_count = os.cpu_count() or 4

    # If user explicitly configured workers, respect it (with bounds checking)
    if config_workers is not None and config_workers > 0:
        if max_workers:
            return min(max(config_workers, min_workers), max_workers)
        return max(config_workers, min_workers)

    # Auto-detect optimal workers
    # Reserve cores for system and other processes
    available_cores = max(cpu_count - reserve_cores, 1)

    # Adjust based on workload type
    if workload_type == "light":
        # Use fewer cores for light workloads
        optimal = max(1, available_cores // 2)
    elif workload_type == "heavy":
        # Use most available cores for heavy workloads
        optimal = available_cores
    else:  # balanced
        # Use 75% of available cores for balanced workloads
        optimal = max(1, int(available_cores * 0.75))

    # Apply bounds
    optimal = max(optimal, min_workers)
    if max_workers:
        optimal = min(optimal, max_workers)

    return optimal


def get_scanner_workers(config) -> int:
    """Get optimal worker count for file scanning.

    File scanning is I/O bound, so we can use more workers.
    """
    configured = getattr(config.scanner, "workers", None) if config else None

    # For I/O bound tasks, we can use more workers
    return get_optimal_workers(
        config_workers=configured,
        min_workers=1,
        max_workers=8,  # Cap at 8 for file scanning
        reserve_cores=2,
        workload_type="light",  # File scanning is relatively light on CPU
    )


def get_ranking_workers(config) -> int:
    """Get optimal worker count for ranking.

    Ranking is CPU bound, so we need to be more conservative.
    """
    configured = getattr(config.ranking, "workers", None) if config else None

    # For CPU bound tasks, be more conservative
    return get_optimal_workers(
        config_workers=configured,
        min_workers=1,
        max_workers=6,  # Cap at 6 for ranking
        reserve_cores=2,
        workload_type="balanced",
    )


def get_analysis_workers(config) -> int:
    """Get optimal worker count for code analysis.

    Code analysis is CPU bound but varies by file size.
    """
    # Check if there's a specific analysis worker config
    configured = None
    if config and hasattr(config, "analysis"):
        configured = getattr(config.analysis, "workers", None)

    return get_optimal_workers(
        config_workers=configured,
        min_workers=1,
        max_workers=4,  # Cap at 4 for analysis
        reserve_cores=2,
        workload_type="balanced",
    )


def log_worker_info(logger, component: str, workers: int, cpu_count: Optional[int] = None):
    """Log worker configuration information.

    Args:
        logger: Logger instance
        component: Component name (e.g., "Scanner", "Ranker")
        workers: Number of workers being used
        cpu_count: Total CPU count (will be detected if not provided)
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1

    # Determine if this is conservative, balanced, or aggressive
    ratio = workers / cpu_count
    if ratio <= 0.25:
        mode = "conservative"
    elif ratio <= 0.5:
        mode = "balanced"
    elif ratio <= 0.75:
        mode = "moderate"
    else:
        mode = "aggressive"

    logger.info(
        f"{component} using {workers} workers out of {cpu_count} CPU cores "
        f"({ratio:.0%} utilization, {mode} mode)"
    )
