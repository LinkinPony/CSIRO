from __future__ import annotations

import os
import queue
import time
import traceback
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp

# Global kill-switch to prevent recursive spawning inside worker processes.
ENV_DISABLE_2GPU_PARALLEL = "CSIRO_DISABLE_2GPU_PARALLEL"


def two_gpu_available() -> bool:
    """
    Return True if the current process can see at least 2 CUDA devices.
    """
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) >= 2
    except Exception:
        return False


def two_gpu_parallel_enabled() -> bool:
    """
    Return True if 2-GPU parallel is available and not disabled via env var.
    """
    if str(os.environ.get(ENV_DISABLE_2GPU_PARALLEL, "")).strip() == "1":
        return False
    return two_gpu_available()


def split_even_odd_indices(n: int) -> Tuple[list[int], list[int]]:
    """
    Split [0..n-1] indices into even and odd positions to balance work across GPUs.
    """
    n = int(n)
    if n <= 0:
        return [], []
    idx0 = list(range(0, n, 2))
    idx1 = list(range(1, n, 2))
    return idx0, idx1


def _terminate_best_effort(proc: mp.Process) -> None:
    try:
        if proc.is_alive():
            proc.terminate()
    except Exception:
        pass


def run_two_processes_spawn(
    *,
    worker: Callable[[mp.Queue, int, int, Any], None],
    payload0: Any,
    payload1: Any,
    device0: int = 0,
    device1: int = 1,
    poll_s: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a 2-way sharded job using torch.multiprocessing spawn and return (res0, res1).

    Worker protocol:
      - worker(q, shard_id, device_id, payload)
      - MUST put a dict into q with keys:
          {"ok": bool, "shard_id": int, ...}
        When ok=False, should include {"error": str, "traceback": str}.
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    p0 = ctx.Process(target=worker, args=(q, 0, int(device0), payload0))
    p1 = ctx.Process(target=worker, args=(q, 1, int(device1), payload1))
    p0.daemon = False
    p1.daemon = False
    p0.start()
    p1.start()

    procs = {0: p0, 1: p1}
    results: Dict[int, Dict[str, Any]] = {}

    try:
        while len(results) < 2:
            try:
                msg = q.get(timeout=max(0.1, float(poll_s)))
            except queue.Empty:
                # If any worker died without sending a message, fail fast.
                for sid, p in procs.items():
                    if sid in results:
                        continue
                    if p.exitcode is not None and int(p.exitcode) != 0:
                        raise RuntimeError(f"2-GPU worker process shard={sid} exited with code {p.exitcode}")
                continue

            if not isinstance(msg, dict):
                raise RuntimeError(f"2-GPU worker returned non-dict message: {type(msg)}")
            sid = msg.get("shard_id", None)
            if sid not in (0, 1):
                raise RuntimeError(f"2-GPU worker returned invalid shard_id: {sid}")
            results[int(sid)] = msg

            # Early failure if a shard reports an error.
            if not bool(msg.get("ok", False)):
                err = str(msg.get("error", "unknown error"))
                tb = str(msg.get("traceback", "")).strip()
                raise RuntimeError(f"2-GPU worker shard={sid} failed: {err}\n{tb}")

        # Ensure clean exit
        for p in procs.values():
            p.join()
        for sid, p in procs.items():
            if p.exitcode is None:
                continue
            if int(p.exitcode) != 0:
                raise RuntimeError(f"2-GPU worker process shard={sid} exited with code {p.exitcode}")
    except Exception:
        # Best-effort terminate the other worker(s) on any failure to avoid hangs.
        for p in procs.values():
            _terminate_best_effort(p)
        for p in procs.values():
            try:
                p.join(timeout=5.0)
            except Exception:
                pass
        raise

    return results[0], results[1]


def worker_guarded(
    q: mp.Queue,
    shard_id: int,
    device_id: int,
    payload: Any,
    *,
    fn: Callable[[Any], Dict[str, Any]],
) -> None:
    """
    Helper to implement worker(...) bodies:
      - sets current CUDA device
      - sets ENV_DISABLE_2GPU_PARALLEL=1 to prevent nested spawns
      - runs fn(payload) and wraps exceptions into a result dict
    """
    os.environ[ENV_DISABLE_2GPU_PARALLEL] = "1"
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(device_id))
            except Exception:
                pass
        out = fn(payload)
        if not isinstance(out, dict):
            raise RuntimeError(f"Worker function must return dict, got {type(out)}")
        msg = dict(out)
        msg["ok"] = True
        msg["shard_id"] = int(shard_id)
        msg["device_id"] = int(device_id)
        q.put(msg)
    except Exception as e:
        tb = traceback.format_exc()
        q.put(
            {
                "ok": False,
                "shard_id": int(shard_id),
                "device_id": int(device_id),
                "error": str(e),
                "traceback": tb,
            }
        )


