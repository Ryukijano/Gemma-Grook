from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VariationalCheckResult:
    is_valid: bool
    estimated_energy: float
    exact_energy: float
    violation_margin: float
    precision_used: Optional[float] = None
    details: Optional[str] = None


def compute_exact_ground_energy(hamiltonian, method: str = "auto") -> float:
    """
    Compute exact ground-state energy using Qrisp's diagonalization or a sparse fallback.
    Pure function; relies only on inputs.
    """
    if method in ("auto", "qrisp"):
        try:
            return float(hamiltonian.ground_state_energy())
        except Exception:
            if method == "qrisp":
                raise
    # Fallback: SciPy sparse eigensolver
    try:
        from scipy.sparse.linalg import eigsh
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for sparse fallback but is not available") from exc

    sparse = hamiltonian.to_sparse_matrix()
    eigvals, _ = eigsh(sparse, k=1, which="SA")
    return float(np.real(eigvals[0]))


def energy_violation(estimated_energy: float, exact_energy: float, *, atol: float = 1e-6, rtol: float = 1e-8) -> bool:
    """
    Return True if the variational bound appears violated: estimate < exact - tol.
    Using both absolute and relative tolerances to account for numeric noise.
    """
    threshold = exact_energy - (atol + abs(exact_energy) * rtol)
    return estimated_energy < threshold


def validate_energy_against_exact(estimated_energy: float, hamiltonian, *, atol: float = 1e-6, rtol: float = 1e-8) -> VariationalCheckResult:
    exact = compute_exact_ground_energy(hamiltonian)
    violated = energy_violation(estimated_energy, exact, atol=atol, rtol=rtol)
    margin = exact - estimated_energy
    return VariationalCheckResult(
        is_valid=not violated,
        estimated_energy=float(estimated_energy),
        exact_energy=float(exact),
        violation_margin=float(margin),
    )


def rerun_with_tighter_precision(
    vqe_problem,
    qarg,
    depth: int,
    *,
    precision_schedule: Iterable[float] = (0.005, 0.0025, 0.001),
    max_iter: int = 500,
    init_type: str = "random",
    optimizer: str = "COBYLA",
    options: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Try re-estimating energy with progressively tighter measurement precision.
    Returns (best_energy, precision_used).
    Pure function w.r.t inputs; caller supplies qarg constructor or instance.
    """
    options = options or {}
    best_energy = float("inf")
    used_precision: Optional[float] = None
    for prec in precision_schedule:
        energy = float(
            vqe_problem.run(
                qarg=qarg,
                depth=depth,
                mes_kwargs={"precision": prec},
                max_iter=max_iter,
                init_type=init_type,
                optimizer=optimizer,
                options=options,
            )
        )
        if energy < best_energy:
            best_energy = energy
            used_precision = prec
    return best_energy, float(used_precision if used_precision is not None else 0.0)


def validate_and_fix(
    vqe_supplier: Callable[[], object],
    hamiltonian,
    qarg_supplier: Callable[[], object],
    depth: int,
    energy_estimate: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-8,
    precision_schedule: Iterable[float] = (0.005, 0.0025),
    max_iter: int = 500,
    init_type: str = "random",
    optimizer: str = "COBYLA",
    options: Optional[dict] = None,
) -> VariationalCheckResult:
    """
    Validate a VQE energy against the exact ground energy and optionally re-run with tighter precision.

    - vqe_supplier: zero-arg function creating a fresh VQEProblem instance
    - qarg_supplier: zero-arg function creating a fresh QuantumVariable or equivalent
    Pure orchestration without side effects outside of VQEProblem.run calls.
    """
    base = validate_energy_against_exact(energy_estimate, hamiltonian, atol=atol, rtol=rtol)
    if base.is_valid:
        return base

    vqe = vqe_supplier()
    qarg = qarg_supplier()
    best_energy, used_prec = rerun_with_tighter_precision(
        vqe,
        qarg,
        depth,
        precision_schedule=precision_schedule,
        max_iter=max_iter,
        init_type=init_type,
        optimizer=optimizer,
        options=options,
    )

    fixed = validate_energy_against_exact(best_energy, hamiltonian, atol=atol, rtol=rtol)
    return VariationalCheckResult(
        is_valid=fixed.is_valid,
        estimated_energy=fixed.estimated_energy,
        exact_energy=fixed.exact_energy,
        violation_margin=fixed.violation_margin,
        precision_used=used_prec,
        details=(
            "accepted after precision tightening"
            if fixed.is_valid
            else "still violates after precision tightening"
        ),
    )


