"""Sparse linear-algebra backend: scipy/SuperLU (CPU) or cuDSS via nvmath (GPU).

Torch tensors are exchanged zero-copy through DLPack. GPU requirements: ``cupy-cuda13x``,
``nvidia-cudss-cu13`` and ``nvmath-python[cu13]`` (torch's own sparse solver is not usable: the
cu130 wheel is built without cuDSS). Benchmarked alternatives (palto e629 mesh, 5k/10k systems):
cupyx spsolve is ~10x slower than SuperLU and inaccurate on the badly scaled SG/hole systems;
cuDSS with the symbolic analysis reused across Newton iterations is 6-20x faster than SuperLU.

On the GPU every system is row-equilibrated (rows divided by their max |entry|): the Poisson and hole
Jacobians have entries ~1e-17..1e-50 and cuDSS treats such pivots as zero otherwise. On the CPU SuperLU
is used as is (equilibration made its pivoting lose accuracy on the p-scaled hole system).
Solvers are cached per ``key`` (one per equation type, whose sparsity pattern is fixed); a pattern
change triggers a new analysis.
"""
from __future__ import annotations

import numpy as np
import torch


class Backend:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.gpu = self.device.type == "cuda"
        self._cache = {}
        self.refine_steps = 2
        self.ruiz = True
        if self.gpu:
            import cupy, cupyx.scipy.sparse as csp
            import nvmath  # noqa: F401  (fail early if cuDSS bindings are missing)
            self.xp, self.sp = cupy, csp
            idx = self.device.index or 0
            cupy.cuda.Device(idx).use()            # cupy arrays / DLPack exchange on torch's device
            torch.cuda.set_device(idx)
        else:
            import scipy.sparse as sp, scipy.sparse.linalg as spla
            self.xp, self.sp, self.spla = np, sp, spla

    def activate(self):
        """Make this backend's device current in the calling thread (CUDA contexts are per thread;
        twins refine from worker threads)."""
        if self.gpu:
            idx = self.device.index or 0
            self.xp.cuda.Device(idx).use()
            torch.cuda.set_device(idx)
            from cuda.core import Device                 # nvmath's cuDSS solvers check cuda.core's context
            Device(idx).set_current()

    # torch <-> backend arrays
    def arr(self, t: torch.Tensor):
        t = t.detach().contiguous()
        return self.xp.from_dlpack(t) if self.gpu else t.cpu().numpy()

    def tens(self, a) -> torch.Tensor:
        return torch.from_dlpack(a) if self.gpu else torch.as_tensor(np.asarray(a), device=self.device)

    def coo(self, rows, cols, vals, shape):
        r, c, v = (self.arr(x) for x in (rows, cols, vals))
        return self.sp.coo_matrix((v, (r, c)), shape=shape).tocsr()

    def solve(self, A, b, key="default", equilibrate=None):
        """Direct solve. Rows are equilibrated on the GPU (cuDSS needs it); on the CPU only if asked, because
        SuperLU's pivoting is more accurate on the unequilibrated hole system (column-scaled by p)."""
        xp = self.xp
        A = A.tocsr()
        A.sum_duplicates(); A.sort_indices()
        dc = None
        if self.gpu and self.ruiz:
            # two-sided (Ruiz) equilibration: the SG continuity and joint Jacobians span ~1e-50..1e+10;
            # row scaling alone left cuDSS solutions with relative residuals up to 1e11
            A, b, dc = self._ruiz(A, b)
        elif self.gpu if equilibrate is None else equilibrate:
            rmax = abs(A).max(axis=1)
            rmax = xp.asarray(rmax.toarray()).ravel() if hasattr(rmax, "toarray") else xp.asarray(rmax).ravel()
            d = 1.0 / xp.clip(xp.where(rmax > 0, rmax, 1.0), 1e-300, None)
            A = self.sp.diags(d) @ A; b = d * b
            A.sort_indices()
        if not self.gpu:
            return self.spla.spsolve(A.tocsc(), b)
        from nvmath.sparse.advanced import DirectSolver
        c = self._cache.get(key)
        same = (c is not None and c["A"].shape == A.shape and c["A"].nnz == A.nnz
                and bool((c["A"].indptr == A.indptr).all()) and bool((c["A"].indices == A.indices).all()))
        if not same:
            if c is not None:
                c["solver"].free()
            Ab, bb = A.copy(), xp.array(b, copy=True)
            s = DirectSolver(Ab, bb); s.plan()
            c = self._cache[key] = dict(A=Ab, b=bb, solver=s)
        else:
            c["A"].data[:] = A.data; c["b"][:] = b
        c["solver"].factorize()
        x = c["solver"].solve().copy()
        # iterative refinement on the factorisation: cuDSS solutions of these badly scaled systems lose
        # several digits (Newton guesses that Sentaurus could not converge from); 2 corrections fix it
        for _ in range(self.refine_steps):
            r = b - c["A"] @ x
            c["b"][:] = r
            x = x + c["solver"].solve()
        return x if dc is None else dc * x

    def _ruiz(self, A, b, sweeps=6):
        xp, sp = self.xp, self.sp
        n = A.shape[0]
        dr, dc = xp.ones(n), xp.ones(A.shape[1])
        M = A
        for _ in range(sweeps):
            r = xp.asarray(abs(M).max(axis=1).toarray()).ravel()
            c = xp.asarray(abs(M).max(axis=0).toarray()).ravel()
            r = 1.0 / xp.sqrt(xp.clip(xp.where(r > 0, r, 1.0), 1e-300, None))
            c = 1.0 / xp.sqrt(xp.clip(xp.where(c > 0, c, 1.0), 1e-300, None))
            M = sp.diags(r) @ M @ sp.diags(c)
            dr *= r; dc *= c
        M = M.tocsr(); M.sort_indices()
        return M, dr * b, dc
