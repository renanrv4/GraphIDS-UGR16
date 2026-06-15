from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


class DynamicGraph:
    def __init__(
        self,
        edge_feature_dim: int,
        feature_names: Optional[List[str]] = None,
        max_edge_age_ms: float = 3600000,
        max_nodes: int = 100000,
        scaler: Optional[MinMaxScaler] = None,
    ):
        self.edge_feature_dim = edge_feature_dim
        self.feature_names = feature_names
        self.max_edge_age_ms = max_edge_age_ms
        self.max_nodes = max_nodes
        self.scaler = scaler

        self.node_map: Dict[str, int] = {}
        self.inverse_node_map: Dict[int, str] = {}
        self._next_node_id: int = 0

        self._edge_src: List[int] = []
        self._edge_dst: List[int] = []
        self._edge_attr: List[List[float]] = []
        self._edge_timestamps: List[float] = []
        self._edge_labels: List[int] = []
        self._edge_active: List[bool] = []

        self.adj_out: Dict[int, List[Tuple[int, int, float]]] = {}
        self.adj_in: Dict[int, List[Tuple[int, int, float]]] = {}

    def get_node_id(self, ip: str) -> int:
        if ip not in self.node_map:
            if len(self.node_map) >= self.max_nodes:
                raise RuntimeError(f"Max nodes limit ({self.max_nodes}) reached")
            node_id = self._next_node_id
            self._next_node_id += 1
            self.node_map[ip] = node_id
            self.inverse_node_map[node_id] = ip
        return self.node_map[ip]

    def has_node(self, ip: str) -> bool:
        return ip in self.node_map

    def add_edge(
        self,
        src_ip: str,
        dst_ip: str,
        edge_attr: List[float],
        timestamp: float,
        label: int = 0,
    ) -> int:
        src_id = self.get_node_id(src_ip)
        dst_id = self.get_node_id(dst_ip)

        eid = len(self._edge_src)
        self._edge_src.append(src_id)
        self._edge_dst.append(dst_id)
        self._edge_attr.append(edge_attr)
        self._edge_timestamps.append(timestamp)
        self._edge_labels.append(label)
        self._edge_active.append(True)

        if src_id not in self.adj_out:
            self.adj_out[src_id] = []
        self.adj_out[src_id].append((dst_id, eid, timestamp))

        if dst_id not in self.adj_in:
            self.adj_in[dst_id] = []
        self.adj_in[dst_id].append((src_id, eid, timestamp))

        return eid

    def add_edges_batch(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        src_col: str,
        dst_col: str,
        feature_cols: List[str],
        label_col: str = "Label",
    ) -> np.ndarray:
        edge_ids = []
        for _, row in df.iterrows():
            eid = self.add_edge(
                row[src_col],
                row[dst_col],
                row[feature_cols].tolist(),
                row[timestamp_col],
                row[label_col],
            )
            edge_ids.append(eid)
        return np.array(edge_ids)

    def _remove_edge_by_id(self, eid: int) -> bool:
        if eid >= len(self._edge_active) or not self._edge_active[eid]:
            return False
        self._edge_active[eid] = False
        src = self._edge_src[eid]
        dst = self._edge_dst[eid]
        ts = self._edge_timestamps[eid]

        if src in self.adj_out:
            self.adj_out[src] = [
                (d, e, t) for d, e, t in self.adj_out[src] if e != eid
            ]
            if not self.adj_out[src]:
                del self.adj_out[src]

        if dst in self.adj_in:
            self.adj_in[dst] = [
                (s, e, t) for s, e, t in self.adj_in[dst] if e != eid
            ]
            if not self.adj_in[dst]:
                del self.adj_in[dst]

        return True

    def remove_stale_edges(
        self,
        current_time: float,
        max_age_ms: Optional[float] = None,
    ) -> int:
        if max_age_ms is None:
            max_age_ms = self.max_edge_age_ms
        cutoff = current_time - max_age_ms
        stale_ids = [
            eid
            for eid in range(len(self._edge_timestamps))
            if self._edge_active[eid] and self._edge_timestamps[eid] < cutoff
        ]
        for eid in stale_ids:
            self._remove_edge_by_id(eid)
        return len(stale_ids)

    def remove_edge(self, edge_id: int) -> bool:
        return self._remove_edge_by_id(edge_id)

    def remove_node(self, node_id: int) -> bool:
        if node_id not in self.inverse_node_map:
            return False
        incident: List[int] = []
        if node_id in self.adj_out:
            for dst, eid, ts in self.adj_out[node_id]:
                incident.append(eid)
        if node_id in self.adj_in:
            for src, eid, ts in self.adj_in[node_id]:
                incident.append(eid)
        for eid in set(incident):
            self._remove_edge_by_id(eid)
        ip = self.inverse_node_map.pop(node_id, None)
        if ip is not None:
            self.node_map.pop(ip, None)
        return True

    def remove_inactive_nodes(
        self,
        max_age_ms: float,
        current_time: float,
    ) -> int:
        cutoff = current_time - max_age_ms
        removed = 0
        for node_id in list(self.get_node_ids()):
            all_old = True
            if node_id in self.adj_out:
                for dst, eid, ts in self.adj_out[node_id]:
                    if ts >= cutoff:
                        all_old = False
                        break
            if all_old and node_id in self.adj_in:
                for src, eid, ts in self.adj_in[node_id]:
                    if ts >= cutoff:
                        all_old = False
                        break
            if all_old:
                self.remove_node(node_id)
                removed += 1
        return removed

    def get_current_graph(self) -> Data:
        active_eids = [
            eid
            for eid in range(len(self._edge_active))
            if self._edge_active[eid]
        ]
        if not active_eids:
            return Data(
                x=torch.zeros(0, self.edge_feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, self.edge_feature_dim),
                edge_labels=torch.zeros(0, dtype=torch.long),
                num_nodes=0,
            )
        node_set: Set[int] = set()
        for eid in active_eids:
            node_set.add(self._edge_src[eid])
            node_set.add(self._edge_dst[eid])
        node_list = sorted(node_set)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        src = [node_to_idx[self._edge_src[eid]] for eid in active_eids]
        dst = [node_to_idx[self._edge_dst[eid]] for eid in active_eids]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(
            [self._edge_attr[eid] for eid in active_eids], dtype=torch.float
        )
        edge_labels = torch.tensor(
            [self._edge_labels[eid] for eid in active_eids], dtype=torch.long
        )
        x = torch.ones(len(node_list), self.edge_feature_dim, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_labels=edge_labels,
            num_nodes=len(node_list),
        )

    def get_snapshot(self, start_time: float, end_time: float) -> Data:
        active_eids = [
            eid
            for eid in range(len(self._edge_timestamps))
            if self._edge_active[eid]
            and start_time <= self._edge_timestamps[eid] <= end_time
        ]
        if not active_eids:
            return Data(
                x=torch.zeros(0, self.edge_feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, self.edge_feature_dim),
                edge_labels=torch.zeros(0, dtype=torch.long),
                num_nodes=0,
            )
        node_set: Set[int] = set()
        for eid in active_eids:
            node_set.add(self._edge_src[eid])
            node_set.add(self._edge_dst[eid])
        node_list = sorted(node_set)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        src = [node_to_idx[self._edge_src[eid]] for eid in active_eids]
        dst = [node_to_idx[self._edge_dst[eid]] for eid in active_eids]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(
            [self._edge_attr[eid] for eid in active_eids], dtype=torch.float
        )
        edge_labels = torch.tensor(
            [self._edge_labels[eid] for eid in active_eids], dtype=torch.long
        )
        x = torch.ones(len(node_list), self.edge_feature_dim, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_labels=edge_labels,
            num_nodes=len(node_list),
        )

    def get_snapshot_dataframe(
        self, start_time: float, end_time: float
    ) -> pd.DataFrame:
        active_eids = [
            eid
            for eid in range(len(self._edge_timestamps))
            if self._edge_active[eid]
            and start_time <= self._edge_timestamps[eid] <= end_time
        ]
        if not active_eids:
            return pd.DataFrame()
        rows = []
        for eid in active_eids:
            row: Dict[str, object] = {
                "IPV4_SRC_ADDR": self.inverse_node_map[self._edge_src[eid]],
                "IPV4_DST_ADDR": self.inverse_node_map[self._edge_dst[eid]],
                "Label": self._edge_labels[eid],
            }
            for j, val in enumerate(self._edge_attr[eid]):
                if self.feature_names is not None and j < len(self.feature_names):
                    row[self.feature_names[j]] = val
                else:
                    row[f"feature_{j}"] = val
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def num_nodes(self) -> int:
        return len(self.node_map)

    @property
    def num_edges(self) -> int:
        return sum(self._edge_active)

    @property
    def num_node_features(self) -> int:
        return self.edge_feature_dim

    @property
    def num_edge_features(self) -> int:
        return self.edge_feature_dim

    def get_node_ids(self) -> List[int]:
        return list(self.inverse_node_map.keys())

    def get_edge_ids_for_node(self, node_id: int) -> List[int]:
        eids: List[int] = []
        if node_id in self.adj_out:
            for dst, eid, ts in self.adj_out[node_id]:
                if self._edge_active[eid]:
                    eids.append(eid)
        if node_id in self.adj_in:
            for src, eid, ts in self.adj_in[node_id]:
                if self._edge_active[eid]:
                    eids.append(eid)
        return eids

    def get_node_degree(self, node_id: int) -> int:
        return len(self.get_edge_ids_for_node(node_id))

    def reset(self):
        self.node_map.clear()
        self.inverse_node_map.clear()
        self._next_node_id = 0
        self._edge_src.clear()
        self._edge_dst.clear()
        self._edge_attr.clear()
        self._edge_timestamps.clear()
        self._edge_labels.clear()
        self._edge_active.clear()
        self.adj_out.clear()
        self.adj_in.clear()

    def fit_scaler(
        self, df: pd.DataFrame, feature_cols: List[str]
    ):
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[feature_cols])

    def transform_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        df = df.copy()
        df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def inverse_transform_features(
        self, features: np.ndarray
    ) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.inverse_transform(features)
