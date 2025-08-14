from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from qrisp import QuantumVariable


def make_trotter_ansatz(hamiltonian, num_steps: int, enforce_sz: bool = True) -> tuple[Callable, int]:
    evol = hamiltonian.trotterization(order=2, method="commuting", forward_evolution=True)

    def ansatz_fn(qv: QuantumVariable, theta):
        assert len(theta) == num_steps
        for t in theta:
            evol(qv, t=t, steps=1)

    return ansatz_fn, num_steps


