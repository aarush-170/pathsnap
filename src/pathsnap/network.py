"""Road-network helpers: connectivity statistics and route checks."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def node_graph(n_nodes: int, arcs: np.ndarray):
    """Directed sparse adjacency matrix over nodes (one entry per arc)."""
    data = np.ones(len(arcs), dtype=np.int8)
    return coo_matrix((data, (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes)).tocsr()


def network_stats(nodes: np.ndarray, arcs: np.ndarray) -> dict:
    n, m = len(nodes), len(arcs)
    valid = (arcs >= 0).all(1) & (arcs < n).all(1)
    a = arcs[valid]
    G = node_graph(n, a)
    n_wcc, wcc_lab = connected_components(G, directed=True, connection="weak")
    n_scc, scc_lab = connected_components(G, directed=True, connection="strong")
    used = np.zeros(n, bool)
    used[a.ravel()] = True
    deg_out = np.bincount(a[:, 0], minlength=n)
    deg_in = np.bincount(a[:, 1], minlength=n)
    # reverse twin: arc (u,v) also present as (v,u) -> two-way street
    key = a[:, 0].astype(np.int64) * (n + 1) + a[:, 1]
    rkey = a[:, 1].astype(np.int64) * (n + 1) + a[:, 0]
    two_way = np.isin(rkey, key).mean() if m else np.nan
    dup = m - len(np.unique(key))
    largest_wcc = np.bincount(wcc_lab[used]).max() / max(used.sum(), 1) if used.any() else np.nan
    largest_scc = np.bincount(scc_lab[used]).max() / max(used.sum(), 1) if used.any() else np.nan
    return dict(
        n_nodes=n, n_arcs=m, invalid_arcs=int((~valid).sum()),
        isolated_nodes=int((~used).sum()), self_loops=int((a[:, 0] == a[:, 1]).sum()),
        duplicate_arcs=int(dup), two_way_frac=float(two_way),
        n_weak_components=int(n_wcc - (~used).sum()),  # exclude isolated singletons
        n_strong_components=int(n_scc - (~used).sum()),
        largest_wcc_frac=float(largest_wcc), largest_scc_frac=float(largest_scc),
        mean_out_degree=float(deg_out[used].mean()) if used.any() else np.nan,
        max_degree=int((deg_out + deg_in).max()) if n else 0,
        dead_end_nodes=int(((deg_out == 0) & used).sum()),
    )


def route_breaks(arcs: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Indices i where route[i] -> route[i+1] is not connected head-to-tail."""
    if route is None or len(route) < 2:
        return np.array([], dtype=int)
    return np.where(arcs[route[:-1], 1] != arcs[route[1:], 0])[0]
