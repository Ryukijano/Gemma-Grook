from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qrisp import QuantumVariable
from qrisp.vqe import VQEProblem

from .features import extract_graph_heisenberg_features
from .quixer import init_quixer_params, quixer_predict
from .ansatz import make_trotter_ansatz
from qrisp_utils import validate_and_fix
from qrisp_utils.spin_operators import apply_spin_penalties
from .quixer_bridge import build_quixer_model, quixer_generate_circuit, tokens_to_qujax_circuit


def run_gqe(
    graph,
    H_problem,
    *,
    J: float,
    B: float,
    steps_max: int = 8,
    seed: int = 0,
    lambda_s2: float = 0.0,
    S_target: float = 0.0,
    lambda_m: float = 0.0,
    M_target: float = 0.0,
    use_quixer: bool = True,
) -> Tuple[float, dict]:
    """
    Minimal GQE pipeline: JAX MLP proposes Trotter step durations; Qrisp builds
    ansatz; VQEProblem optimizes remaining degrees. Returns best energy and info.
    """
    features = extract_graph_heisenberg_features(graph, J, B)
    x = jnp.asarray(features)

    input_dim = x.shape[0]
    hidden_dim = max(16, input_dim * 2)
    output_dim = steps_max

    key = jax.random.PRNGKey(seed)
    params = init_quixer_params(key, input_dim, hidden_dim, output_dim)

    init_ts = jnp.asarray(quixer_predict(params, x))
    # Normalize durations so their sum stays moderate
    init_ts = (init_ts / (1e-6 + jnp.sum(init_ts))) * 1.0

    # Option A: Quixer+qujax-generated circuit (converted wrapper)
    use_quixer = bool(use_quixer)
    if use_quixer:
        model = build_quixer_model(graph.number_of_nodes(), depth=steps_max, key=key)
        toks = quixer_generate_circuit(model, x, key=key)
        qx_circ, num_params = tokens_to_qujax_circuit(toks["tokens"], graph.number_of_nodes())

        def ansatz_fn(qv: QuantumVariable, theta):
            _ = theta
            return qv
    else:
        # Build Trotter ansatz with steps_max layers
        ansatz_fn, num_params = make_trotter_ansatz(H_problem, steps_max)

    H_eff = apply_spin_penalties(
        H_problem,
        num_qubits=graph.number_of_nodes(),
        lambda_s2=lambda_s2,
        S_target=S_target,
        lambda_m=lambda_m,
        M_target=M_target,
    )

    vqe = VQEProblem(
        hamiltonian=H_eff,
        ansatz_function=ansatz_fn,
        num_params=num_params,
        callback=False,
    )

    qv = QuantumVariable(graph.number_of_nodes())

    # Use init_ts as starting params
    energy = float(
        vqe.run(
            qarg=qv,
            depth=1,
            mes_kwargs={"precision": 0.005},
            max_iter=500,
            init_type="custom",
            init_params=np.asarray(np.clip(np.array(init_ts), 0.0, 2.0), dtype=float),
            optimizer="COBYLA",
            options={},
        )
    )

    # Variational guard with precision tightening
    res = validate_and_fix(
        vqe_supplier=lambda: vqe,
        hamiltonian=H_eff,
        qarg_supplier=lambda: QuantumVariable(graph.number_of_nodes()),
        depth=1,
        energy_estimate=energy,
        precision_schedule=(0.005, 0.0025, 0.001),
        max_iter=500,
        optimizer="COBYLA",
    )

    info = {
        "proposed_trotter_times": np.asarray(init_ts).tolist(),
        "valid": res.is_valid,
        "exact_energy": res.exact_energy,
        "precision_used": res.precision_used,
        "details": res.details,
        "used_quixer": use_quixer,
    }
    return float(res.estimated_energy), info


