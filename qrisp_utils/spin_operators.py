from __future__ import annotations

from typing import Tuple

import numpy as np


def build_sz_terms(num_qubits: int):
    from qrisp import Z
    return [Z(i) for i in range(num_qubits)]


def build_s2_operator(num_qubits: int):
    from qrisp import X, Y, Z
    # For spin-1/2 chain: S^2 = 3/4 N + 1/2 sum_{i<j} (X_i X_j + Y_i Y_j + Z_i Z_j)
    N = num_qubits
    s2 = 0.75 * N
    for i in range(N):
        for j in range(i + 1, N):
            s2 = s2 + 0.5 * (X(i) * X(j) + Y(i) * Y(j) + Z(i) * Z(j))
    return s2


def apply_spin_penalties(H, *, num_qubits: int, lambda_s2: float = 0.0, S_target: float = 0.0, lambda_m: float = 0.0, M_target: float = 0.0):
    """
    Return effective Hamiltonian H + λS2 (S^2 - S_t(S_t+1))^2 + λM (S_z - M_t)^2.
    """
    if lambda_s2 == 0.0 and lambda_m == 0.0:
        return H

    H_eff = H
    if lambda_m != 0.0:
        from qrisp import Z
        # S_z = 1/2 sum Z_i
        sz = 0
        for i in range(num_qubits):
            sz = sz + 0.5 * Z(i)
        H_eff = H_eff + lambda_m * (sz - M_target) * (sz - M_target)

    if lambda_s2 != 0.0:
        s2 = build_s2_operator(num_qubits)
        st = S_target * (S_target + 1.0)
        H_eff = H_eff + lambda_s2 * (s2 - st) * (s2 - st)

    return H_eff


