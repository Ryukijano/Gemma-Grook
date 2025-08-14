from __future__ import annotations

from typing import Dict
import networkx as nx

import numpy as np


def extract_graph_heisenberg_features(graph: nx.Graph, J: float, B: float) -> np.ndarray:
    """
    Simple, deterministic features from a small Heisenberg graph instance.
    This is intentionally lightweight and side-effect free.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degrees = np.array([deg for _, deg in graph.degree()], dtype=np.float32)
    degree_stats = np.array(
        [degrees.min(), degrees.max(), degrees.mean(), degrees.std() if degrees.size > 1 else 0.0],
        dtype=np.float32,
    )
    features = np.array(
        [
            float(num_nodes),
            float(num_edges),
            float(J),
            float(B),
        ],
        dtype=np.float32,
    )
    clustering = nx.average_clustering(graph) if num_nodes > 1 else 0.0
    spectral_gap = 0.0
    try:
        # Laplacian spectral gap (λ2) for connectivity/correlation proxy
        lap = nx.normalized_laplacian_matrix(graph).astype(float)
        evals = np.linalg.eigvalsh(lap.A)
        evals.sort()
        if evals.size >= 2:
            spectral_gap = float(evals[1])
    except Exception:
        pass

    extra = np.array([clustering, spectral_gap], dtype=np.float32)
    return np.concatenate([features, degree_stats, extra], axis=0)


