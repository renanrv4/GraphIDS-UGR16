import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from models.graphids import GraphIDS
from utils.dynamic_graph import DynamicGraph


class SlidingWindowBuffer:
    def __init__(self, window_size: int, embedding_dim: int):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self._buffer: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, embedding: torch.Tensor) -> None:
        with self._lock:
            self._buffer.append(embedding.detach().clone())

    def get_window(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            n = len(self._buffer)
            emb = torch.zeros(self.window_size, self.embedding_dim)
            mask = torch.zeros(self.window_size, self.embedding_dim, dtype=torch.bool)
            if n > 0:
                start = self.window_size - n
                emb[start:] = torch.stack(list(self._buffer))
                mask[start:] = True
            return emb.unsqueeze(0), mask.unsqueeze(0)

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


class AdaptiveThreshold:
    def __init__(
        self,
        method: str = "mad",
        multiplier: float = 10.0,
        window_size: int = 1000,
        min_window: int = 100,
    ):
        self.method = method
        self.multiplier = multiplier
        self.window_size = window_size
        self.min_window = min_window
        self._scores: List[float] = []
        self._labels: List[Optional[int]] = []
        self._threshold: Optional[float] = None
        self._lock = threading.Lock()

    def add_score(self, score: float, label: Optional[int] = None) -> None:
        with self._lock:
            self._scores.append(score)
            self._labels.append(label)
            if len(self._scores) > self.window_size:
                self._scores.pop(0)
                self._labels.pop(0)

    def compute(self) -> float:
        with self._lock:
            if len(self._scores) < self.min_window:
                self._threshold = float("inf")
                return self._threshold

            scores = np.array(self._scores)

            if self.method == "mad":
                median = float(np.median(scores))
                mad = float(np.median(np.abs(scores - median)))
                if mad == 0.0:
                    mad = float(np.std(scores)) if len(scores) > 1 else 1.0
                self._threshold = median + self.multiplier * mad

            elif self.method == "validation_f1":
                labeled = [
                    (s, l)
                    for s, l in zip(self._scores, self._labels)
                    if l is not None
                ]
                if len(labeled) < self.min_window:
                    self._threshold = float("inf")
                    return self._threshold
                scores_arr = np.array([s for s, l in labeled])
                labels_arr = np.array([l for s, l in labeled])
                if len(np.unique(labels_arr)) < 2:
                    self._threshold = float("inf")
                    return self._threshold

                best_f1 = 0.0
                best_th = float("inf")
                candidates = np.percentile(
                    scores_arr, np.linspace(50, 99.9, 100)
                )
                for th in candidates:
                    pred = (scores_arr > th).astype(int)
                    tp = np.sum((pred == 1) & (labels_arr == 1))
                    fp = np.sum((pred == 1) & (labels_arr == 0))
                    fn = np.sum((pred == 0) & (labels_arr == 1))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th
                self._threshold = float(best_th)

            else:
                raise ValueError(f"Unknown threshold method: {self.method}")

            return self._threshold

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            return float("inf")
        return self._threshold

    def reset(self) -> None:
        with self._lock:
            self._scores.clear()
            self._labels.clear()
            self._threshold = None


class StreamingDetector:
    def __init__(
        self,
        model: GraphIDS,
        dynamic_graph: DynamicGraph,
        window_size: int = 512,
        embedding_dim: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold_method: str = "mad",
        threshold_multiplier: float = 10.0,
        adaptive_threshold: bool = True,
        adaptation_window: int = 1000,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dynamic_graph = dynamic_graph
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.adaptive_threshold = adaptive_threshold

        self.buffer = SlidingWindowBuffer(window_size, embedding_dim)
        self.thresholder = AdaptiveThreshold(
            method=threshold_method,
            multiplier=threshold_multiplier,
            window_size=adaptation_window,
            min_window=max(100, adaptation_window // 10),
        )

        self.total_flows: int = 0
        self.total_alerts: int = 0
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()

    def process_flow(
        self,
        src_ip: str,
        dst_ip: str,
        edge_features: List[float],
        timestamp: float,
        label: Optional[int] = None,
    ) -> Tuple[float, bool]:
        with self._process_lock:
            with torch.no_grad():
                self.dynamic_graph.add_edge(
                    src_ip, dst_ip, edge_features, timestamp, label or 0
                )

                data = self.dynamic_graph.get_current_graph()
                if data.num_nodes == 0 or data.edge_index.shape[1] == 0:
                    return 0.0, False

                data = data.to(self.device)
                src_idx = int(data.edge_index[0, -1].item())
                dst_idx = int(data.edge_index[1, -1].item())
                edge_couples = torch.tensor(
                    [[src_idx, dst_idx]], device=self.device
                )

                embedding = self.model.encoder(
                    data.edge_index,
                    data.edge_attr,
                    edge_couples,
                    data.num_nodes,
                )

                self.buffer.add(embedding.squeeze(0).cpu())

                anomaly_score = 0.0
                if self.buffer.is_ready():
                    window, mask = self.buffer.get_window()
                    window = window.to(self.device)
                    mask = mask.to(self.device)

                    reconstructed = self.model.transformer(
                        window, padding_mask=mask
                    )
                    valid = mask[0, :, 0].bool()
                    diff = (window[0] - reconstructed[0]) ** 2
                    anomaly_score = float(diff[valid].mean().item())

                self.thresholder.add_score(anomaly_score, label)
                if self.adaptive_threshold:
                    self.thresholder.compute()

                current_th = self.thresholder.threshold
                is_anomalous = anomaly_score > current_th

                with self._lock:
                    self.total_flows += 1
                    if is_anomalous:
                        self.total_alerts += 1

                return anomaly_score, is_anomalous

    def process_batch(
        self,
        flows_df: pd.DataFrame,
        timestamp_col: str = "FLOW_START_MILLISECONDS",
        src_col: str = "IPV4_SRC_ADDR",
        dst_col: str = "IPV4_DST_ADDR",
        feature_cols: Optional[List[str]] = None,
        label_col: str = "Label",
    ) -> List[Tuple[float, bool]]:
        if feature_cols is None:
            feature_cols = [
                c
                for c in flows_df.columns
                if c
                not in {timestamp_col, src_col, dst_col, label_col}
            ]

        with self._process_lock:
            with torch.no_grad():
                self.dynamic_graph.add_edges_batch(
                    flows_df,
                    timestamp_col,
                    src_col,
                    dst_col,
                    feature_cols,
                    label_col,
                )
                n_new = len(flows_df)

                data = self.dynamic_graph.get_current_graph()
                if data.num_nodes == 0 or data.edge_index.shape[1] == 0:
                    return [(0.0, False)] * n_new

                data = data.to(self.device)
                total_edges = data.edge_index.shape[1]

                src_indices = data.edge_index[0, -n_new:].cpu().tolist()
                dst_indices = data.edge_index[1, -n_new:].cpu().tolist()
                edge_couples = torch.tensor(
                    list(zip(src_indices, dst_indices)), device=self.device
                )

                embeddings = self.model.encoder(
                    data.edge_index,
                    data.edge_attr,
                    edge_couples,
                    data.num_nodes,
                )

                windows = []
                masks = []
                for i in range(n_new):
                    self.buffer.add(embeddings[i].cpu())
                    win, mask = self.buffer.get_window()
                    windows.append(win)
                    masks.append(mask)

                labels_list = (
                    flows_df[label_col].tolist()
                    if label_col in flows_df.columns
                    else [None] * n_new
                )

                if not windows:
                    return [(0.0, False)] * n_new

                windows = torch.cat(windows, dim=0).to(self.device)
                masks = torch.cat(masks, dim=0).to(self.device)

                reconstructed = self.model.transformer(
                    windows, padding_mask=masks
                )

                valid = masks.any(dim=-1)
                diff = (windows - reconstructed) ** 2
                valid_3d = valid.unsqueeze(-1)
                masked_diff = diff * valid_3d
                sum_diff = masked_diff.sum(dim=(1, 2))
                num_valid = valid_3d.sum(dim=(1, 2))
                errors = (sum_diff / num_valid).tolist()

                results = []
                for i in range(n_new):
                    score = errors[i]
                    lbl = (
                        labels_list[i]
                        if i < len(labels_list)
                        else None
                    )

                    self.thresholder.add_score(score, lbl)
                    if self.adaptive_threshold:
                        self.thresholder.compute()

                    is_anomalous = (
                        score > self.thresholder.threshold
                    )

                    with self._lock:
                        self.total_flows += 1
                        if is_anomalous:
                            self.total_alerts += 1

                    results.append((score, is_anomalous))

                return results

    def update_threshold(self, force: bool = False) -> float:
        if self.adaptive_threshold or force:
            return self.thresholder.compute()
        return self.thresholder.threshold

    def get_recent_scores(
        self, n: Optional[int] = None
    ) -> torch.Tensor:
        if n is None:
            n = self.thresholder.window_size
        with self.thresholder._lock:
            scores = self.thresholder._scores[-n:]
        return torch.tensor(scores)

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            alert_rate = (
                self.total_alerts / self.total_flows
                if self.total_flows > 0
                else 0.0
            )
            stats = {
                "total_flows": self.total_flows,
                "total_alerts": self.total_alerts,
                "alert_rate": alert_rate,
                "current_threshold": self.thresholder.threshold,
                "buffer_fill": self.buffer.size,
                "method": self.thresholder.method,
                "adaptive_threshold": self.adaptive_threshold,
            }
        return stats

    def reset(self):
        self.buffer.clear()
        self.thresholder.reset()
        with self._lock:
            self.total_flows = 0
            self.total_alerts = 0
