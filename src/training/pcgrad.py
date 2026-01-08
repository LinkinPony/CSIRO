from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class PCGradConfig:
    """
    Configuration for PCGrad (Projected Conflicting Gradients).

    Notes:
    - This implementation operates on a list of per-task gradient lists (task -> param -> grad tensor/None).
    - It is intentionally framework-agnostic; Lightning integration lives in the model's backward hook.
    """

    enabled: bool = False
    eps: float = 1.0e-8
    reduction: str = "sum"  # "sum" | "mean"
    shuffle_tasks: bool = True
    seed: Optional[int] = None


def _flatten_grads_with_masks(
    grads_per_task: Sequence[Sequence[Optional[Tensor]]],
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Tensor,  # g_mat (T, D)
    Tensor,  # m_mat (T, D)
    List[int],  # active_idx in original param list
    List[int],  # offsets (len = A+1)
    List[torch.Size],  # shapes per active param
    List[torch.dtype],  # dtypes per active param
]:
    """
    Flatten a per-task list of per-parameter gradients into:
      - g_mat: (T, D) dense gradient matrix (None -> 0)
      - m_mat: (T, D) support mask matrix (grad exists -> 1 else 0)

    We keep "support" semantics (None != zero): the mask is used by PCGrad to avoid
    injecting gradients into parameters that were unused by a task.
    """
    T = int(len(grads_per_task))
    if T <= 0:
        raise ValueError("_flatten_grads_with_masks: empty grads_per_task")
    P = int(len(grads_per_task[0]))
    if any(len(g) != P for g in grads_per_task):
        raise ValueError("_flatten_grads_with_masks: inconsistent per-task gradient lengths.")

    active_idx: List[int] = []
    shapes: List[torch.Size] = []
    dtypes: List[torch.dtype] = []
    devices: List[torch.device] = []
    numels: List[int] = []

    for k in range(P):
        g0 = None
        for i in range(T):
            gi = grads_per_task[i][k]
            if isinstance(gi, Tensor):
                g0 = gi
                break
        if g0 is None:
            continue
        active_idx.append(int(k))
        shapes.append(g0.shape)
        dtypes.append(g0.dtype)
        devices.append(g0.device)
        numels.append(int(g0.numel()))

    if not active_idx:
        # No gradients exist at all.
        g_mat = torch.zeros((T, 0), dtype=compute_dtype)
        m_mat = torch.zeros((T, 0), dtype=compute_dtype)
        return g_mat, m_mat, [], [0], [], []

    device0 = devices[0]
    if any(d != device0 for d in devices):
        raise ValueError("_flatten_grads_with_masks: gradients span multiple devices; unsupported.")

    offsets: List[int] = [0]
    for n in numels:
        offsets.append(offsets[-1] + int(n))
    total_dim = int(offsets[-1])

    g_rows: List[Tensor] = []
    m_rows: List[Tensor] = []
    for i in range(T):
        v = torch.zeros(total_dim, device=device0, dtype=compute_dtype)
        m = torch.zeros(total_dim, device=device0, dtype=compute_dtype)
        for a, k in enumerate(active_idx):
            gi = grads_per_task[i][k]
            if not isinstance(gi, Tensor):
                continue
            s = int(offsets[a])
            e = int(offsets[a + 1])
            v[s:e] = gi.reshape(-1).to(dtype=compute_dtype)
            m[s:e] = 1.0
        g_rows.append(v)
        m_rows.append(m)
    g_mat = torch.stack(g_rows, dim=0)
    m_mat = torch.stack(m_rows, dim=0)
    return g_mat, m_mat, active_idx, offsets, shapes, dtypes


def _unflatten_grad_vector(
    out_vec: Tensor,
    *,
    P: int,
    active_idx: List[int],
    offsets: List[int],
    shapes: List[torch.Size],
    dtypes: List[torch.dtype],
) -> List[Optional[Tensor]]:
    out: List[Optional[Tensor]] = [None for _ in range(P)]
    for a, k in enumerate(active_idx):
        s = int(offsets[a])
        e = int(offsets[a + 1])
        seg = out_vec[s:e].view(shapes[a]).to(dtype=dtypes[a])
        out[k] = seg
    return out


def pcgrad_project(
    grads_per_task: Sequence[Sequence[Optional[Tensor]]],
    *,
    eps: float = 1.0e-8,
    reduction: str = "sum",
    shuffle_tasks: bool = True,
    seed: Optional[int] = None,
) -> List[Optional[Tensor]]:
    """
    Apply PCGrad projection to a set of per-task gradients.

    Args:
        grads_per_task: list of T tasks, each a list of P gradients (one per parameter).
                        Gradients can be None (unused).
        eps: numeric stability epsilon for projection denominator.
        reduction: "sum" or "mean" across tasks.
        shuffle_tasks: if True, use a random task ordering for projections (PCGrad paper default).
        seed: optional RNG seed used when shuffling tasks.

    Returns:
        A list of length P with the combined gradient per parameter (Tensor or None).
    """
    T = int(len(grads_per_task))
    if T <= 0:
        return []
    P = int(len(grads_per_task[0]))
    if any(len(g) != P for g in grads_per_task):
        raise ValueError("pcgrad_project: inconsistent per-task gradient lengths.")

    compute_dtype = torch.float32
    g_mat, m_mat, active_idx, offsets, shapes, dtypes = _flatten_grads_with_masks(
        grads_per_task, compute_dtype=compute_dtype
    )
    if g_mat.numel() == 0:
        return [None for _ in range(P)]
    device0 = g_mat.device

    # RNG for shuffling task order.
    gen: Optional[torch.Generator] = None
    if shuffle_tasks:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

    # Project each task vector against others when they conflict.
    proj_vecs: List[Tensor] = []
    for i in range(T):
        g = g_mat[i].clone()
        tmp = torch.empty_like(g)  # reuse buffer to avoid per-(i,j) allocations

        # IMPORTANT: replicate the repo's original semantics:
        # dot/denom are computed ONLY over coordinates where task i has gradients (and task j too).
        # This prevents projecting into task-i "absent" parameters (which were None) and keeps
        # PCGrad surgery restricted to shared-support coordinates.
        mi = m_mat[i]  # (D,)
        masked_mat = g_mat * mi  # (T, D)  (also implicitly masks out task-j absent coords via zeros in g_mat[j])
        denom_i = (masked_mat * masked_mat).sum(dim=1) + float(eps)  # (T,)

        if shuffle_tasks:
            order = torch.randperm(T, generator=gen).tolist()  # type: ignore[arg-type]
        else:
            order = list(range(T))
        for j in order:
            if j == i:
                continue
            gj = masked_mat[j]
            d = torch.dot(g, gj)
            # coeff = d/denom if d<0 else 0  -> coeff = clamp_max(d/denom, 0)
            coeff = torch.clamp(d / denom_i[j], max=0.0)
            # g <- g - coeff * gj  (in-place; coeff is scalar tensor)
            torch.mul(gj, coeff, out=tmp)
            g.sub_(tmp)
        proj_vecs.append(g)

    # Combine projected gradients across tasks into a single flattened vector.
    out_vec = torch.zeros(g_mat.shape[1], device=device0, dtype=compute_dtype)
    for g in proj_vecs:
        out_vec.add_(g)

    red = str(reduction or "sum").lower().strip()
    if red == "mean" and T > 0:
        out_vec.div_(float(T))

    return _unflatten_grad_vector(
        out_vec,
        P=P,
        active_idx=active_idx,
        offsets=offsets,
        shapes=shapes,
        dtypes=dtypes,
    )


def pcgrad_project_primary_anchored(
    *,
    primary_grads: Sequence[Optional[Tensor]],
    aux_grads_per_task: Sequence[Sequence[Optional[Tensor]]],
    primary_count: int = 1,
    eps: float = 1.0e-8,
    reduction: str = "sum",
    shuffle_tasks: bool = True,
    seed: Optional[int] = None,
) -> List[Optional[Tensor]]:
    """
    Primary-anchored gradient surgery:

    - Compute a "primary" anchor gradient g_p (provided as `primary_grads`, typically the sum of primary tasks)
    - For each auxiliary task gradient g_a:
        if <g_a, g_p> < 0, project g_a to remove the conflicting component along g_p
    - Return combined gradient: g = g_p + sum(projected g_a)

    Support/None semantics:
    - Projection and subtraction are restricted to parameters used by the auxiliary task (avoid injecting
      gradients into parameters that were unused by that auxiliary task).
    """
    # Build a combined list with primary as task 0 and aux tasks as tasks 1..A for flattening.
    tasks: List[List[Optional[Tensor]]] = [list(primary_grads)] + [list(x) for x in aux_grads_per_task]
    T = int(len(tasks))
    if T <= 0:
        return []
    P = int(len(tasks[0]))
    if any(len(g) != P for g in tasks):
        raise ValueError("pcgrad_project_primary_anchored: inconsistent gradient lengths.")

    compute_dtype = torch.float32
    g_mat, m_mat, active_idx, offsets, shapes, dtypes = _flatten_grads_with_masks(
        tasks, compute_dtype=compute_dtype
    )
    if g_mat.numel() == 0:
        return [None for _ in range(P)]
    device0 = g_mat.device

    g_primary = g_mat[0]  # (D,)

    # RNG for shuffling aux order (optional; changes the result slightly).
    gen: Optional[torch.Generator] = None
    if shuffle_tasks:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

    # Accumulate: start with primary, then add projected aux.
    out_vec = g_primary.clone()
    tmp = torch.empty_like(out_vec)

    aux_indices = list(range(1, T))
    if shuffle_tasks and len(aux_indices) > 1:
        perm = torch.randperm(len(aux_indices), generator=gen).tolist()  # type: ignore[arg-type]
        aux_indices = [aux_indices[i] for i in perm]

    for idx in aux_indices:
        g_aux = g_mat[idx].clone()
        m_aux = m_mat[idx]  # (D,)
        # Restrict anchor to aux support (and implicitly to primary support via zeros in g_primary).
        gp_masked = g_primary * m_aux
        denom = torch.dot(gp_masked, gp_masked) + float(eps)
        if not (denom.abs() > 0):
            out_vec.add_(g_aux)
            continue
        dot = torch.dot(g_aux, gp_masked)
        coeff = torch.clamp(dot / denom, max=0.0)
        torch.mul(gp_masked, coeff, out=tmp)
        g_aux.sub_(tmp)
        out_vec.add_(g_aux)

    # Mean scaling over "effective task count" (primary_count + number of aux tasks).
    red = str(reduction or "sum").lower().strip()
    if red == "mean":
        denom_tasks = int(max(1, int(primary_count) + int(len(aux_grads_per_task))))
        out_vec.div_(float(denom_tasks))

    return _unflatten_grad_vector(
        out_vec,
        P=P,
        active_idx=active_idx,
        offsets=offsets,
        shapes=shapes,
        dtypes=dtypes,
    )


