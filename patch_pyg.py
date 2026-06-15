"""
Monkey-patch PyG's NeighborSampler to work without pyg-lib or torch-sparse.
Usado em sistemas com GLIBC < 2.32 (ex: Ubuntu 20.04).
"""

import torch
import torch_geometric.typing


def _neighbor_sample(
    colptr: torch.Tensor,
    row: torch.Tensor,
    seed: torch.Tensor,
    num_neighbors: list,
    replace: bool,
    subgraph_type: bool,
):
    """Pure-PyTorch neighbor sampler for num_neighbors=[-1] (all neighbors).
    Equivalente funcional a torch.ops.torch_sparse.neighbor_sample,
    mas sem dependência C++.
    """
    if isinstance(num_neighbors, (list, tuple)):
        num_neighbors = torch.tensor(num_neighbors, dtype=torch.long)

    seed = seed.to(colptr.device)

    unique_seed = seed.unique()
    sampled_nodes = set(unique_seed.tolist())
    seed_set = set(unique_seed.tolist())

    edge_rows, edge_cols, edge_ids = [], [], []
    edge_id_offset = 0

    for hop_idx in range(num_neighbors.numel()):
        n_neigh = int(num_neighbors[hop_idx])
        if n_neigh == 0:
            break

        nodes_this_hop = list(seed_set)
        new_nodes = []

        for node_id in nodes_this_hop:
            start = int(colptr[node_id])
            end = int(colptr[node_id + 1])
            if start >= end:
                continue

            neighs = row[start:end]
            eids = torch.arange(start, end, device=row.device)

            if n_neigh != -1:
                perm = torch.randperm(len(neighs), device=row.device)
                neighs = neighs[perm[:n_neigh]]
                eids = eids[perm[:n_neigh]]

            dst = node_id
            for n_idx in range(len(neighs)):
                src = int(neighs[n_idx])
                edge_rows.append(src)
                edge_cols.append(dst)
                eid_val = int(eids[n_idx])
                edge_ids.append(eid_val)
                if src not in sampled_nodes:
                    sampled_nodes.add(src)
                    new_nodes.append(src)

        seed_set = set(new_nodes)

    all_nodes = sorted(sampled_nodes)
    node_tensor = torch.tensor(all_nodes, dtype=torch.long, device=colptr.device)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    row_out = torch.tensor(
        [node_to_idx[r] for r in edge_rows],
        dtype=torch.long, device=colptr.device,
    )
    col_out = torch.tensor(
        [node_to_idx[c] for c in edge_cols],
        dtype=torch.long, device=colptr.device,
    )
    edge_out = torch.tensor(edge_ids, dtype=torch.long, device=colptr.device)

    return (node_tensor, row_out, col_out, edge_out)


class TorchSparseOps:
    class neighbor_sample:
        @staticmethod
        def forward(colptr, row, seed, num_neighbors, replace, subgraph_type):
            return _neighbor_sample(
                colptr, row, seed, num_neighbors, replace, subgraph_type
            )


def apply():
    """Aplica o patch: registra neighbor_sample e marca WITH_TORCH_SPARSE como True."""
    torch_geometric.typing.WITH_PYG_LIB = False
    torch_geometric.typing.WITH_TORCH_SPARSE = True

    if not hasattr(torch.ops, "torch_sparse"):
        class TorchSparse:
            class neighbor_sample:
                pass

        torch.ops.torch_sparse = TorchSparse()

    torch.ops.torch_sparse.neighbor_sample = _neighbor_sample

    import torch_geometric.sampler.neighbor_sampler as ns
    import torch_geometric.loader.link_loader as ll

    print("[patch_pyg] NeighborSampler patched: using pure-PyTorch fallback")
