from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def init_quixer_params(key: jax.Array, input_dim: int, hidden_dim: int, output_dim: int) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (input_dim, hidden_dim)) * (1.0 / jnp.sqrt(input_dim))
    b1 = jnp.zeros((hidden_dim,))
    W2 = jax.random.normal(k2, (hidden_dim, output_dim)) * (1.0 / jnp.sqrt(hidden_dim))
    b2 = jnp.zeros((output_dim,))
    return W1, b1, W2, b2


@jax.jit
def quixer_predict(params, x: jnp.ndarray) -> jnp.ndarray:
    W1, b1, W2, b2 = params
    h = jnp.tanh(x @ W1 + b1)
    y = jnp.softplus(h @ W2 + b2)
    return y


