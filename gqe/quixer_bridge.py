from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import qujax as qx
    import quixer as qxr
except Exception as exc:  # pragma: no cover
    qx = None
    qxr = None


def build_quixer_model(num_qubits: int, depth: int, *, key: jax.Array) -> Any:
    """
    Construct a simple Quixer transformer that outputs circuit tokens for qujax.
    This is a placeholder wiring assuming Quixer exposes a token generator API.
    """
    if qxr is None:
        raise RuntimeError("Quixer not installed. Run scripts/bootstrap_quixer.sh")

    model = qxr.models.TransformerAnsatz(
        num_qubits=num_qubits,
        depth=depth,
        d_model=128,
        num_heads=4,
        mlp_dim=256,
        dropout_rate=0.0,
        key=key,
    )
    return model


def quixer_generate_circuit(model: Any, features: jnp.ndarray, *, key: jax.Array) -> Dict[str, Any]:
    """
    Use Quixer to sample a circuit program/token sequence and convert to qujax circuit.
    """
    tokens = model.sample(features, key=key)
    return {"tokens": tokens}


def tokens_to_qujax_circuit(tokens: Any, num_qubits: int) -> Tuple[qx.Circuit, int]:
    """
    Convert transformer tokens to a qujax circuit and return (circuit, num_params).
    This assumes tokens include parameterized gates that map to qujax gates.
    """
    if qx is None:
        raise RuntimeError("qujax not installed. Run scripts/bootstrap_quixer.sh")

    # Placeholder: build a simple RX-RZ entangling ladder depending on tokens length
    layers = max(1, len(tokens) // max(1, num_qubits)) if hasattr(tokens, "__len__") else 2

    param_count = 2 * num_qubits * layers
    def layer(params, state):
        idx = 0
        for _ in range(layers):
            for q in range(num_qubits):
                state = qx.rx(params[idx])(state, q); idx += 1
                state = qx.rz(params[idx])(state, q); idx += 1
            for q in range(num_qubits - 1):
                state = qx.cx()(state, q, q + 1)
        return state

    circ = qx.Circuit(num_qubits, layer)
    return circ, param_count


