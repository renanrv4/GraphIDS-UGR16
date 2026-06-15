import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from inference.streaming import AdaptiveThreshold, SlidingWindowBuffer, StreamingDetector
from utils.dynamic_graph import DynamicGraph


class MockEncoder(torch.nn.Module):
    def forward(self, edge_index, edge_attr, edge_couples, num_nodes,
                temporal_weights=None, node_mask=None):
        batch_size = edge_couples.shape[0]
        return torch.rand(batch_size, 8)


class MockTransformer(torch.nn.Module):
    def forward(self, src, padding_mask=None):
        return torch.rand_like(src)


class MockGraphIDS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MockEncoder()
        self.transformer = MockTransformer()

    def to(self, device):
        return self

    def eval(self):
        return self


class TestSlidingWindowBuffer:
    def test_is_ready_returns_false_when_empty(self):
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        assert not buf.is_ready()

    def test_is_ready_returns_true_after_adding(self):
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        buf.add(torch.rand(8))
        assert buf.is_ready()

    def test_get_window_returns_correct_shape(self):
        torch.manual_seed(0)
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        buf.add(torch.rand(8))
        buf.add(torch.rand(8))
        emb, mask = buf.get_window()
        assert emb.shape == (1, 4, 8)
        assert mask.shape == (1, 4, 8)
        assert mask.dtype == torch.bool

    def test_get_window_mask_has_correct_positions(self):
        torch.manual_seed(0)
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        buf.add(torch.rand(8))
        buf.add(torch.rand(8))
        emb, mask = buf.get_window()
        assert not mask[0, 0].any()
        assert not mask[0, 1].any()
        assert mask[0, 2].all()
        assert mask[0, 3].all()

    def test_buffer_wraps_on_overflow(self):
        torch.manual_seed(0)
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        for _ in range(6):
            buf.add(torch.rand(8))
        assert buf.size == 4
        emb, mask = buf.get_window()
        assert mask[0].all()

    def test_clear_empties_buffer(self):
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        buf.add(torch.rand(8))
        buf.clear()
        assert not buf.is_ready()
        assert buf.size == 0

    def test_size_property(self):
        buf = SlidingWindowBuffer(window_size=4, embedding_dim=8)
        assert buf.size == 0
        buf.add(torch.rand(8))
        assert buf.size == 1
        buf.add(torch.rand(8))
        assert buf.size == 2


class TestAdaptiveThreshold:
    def test_threshold_returns_inf_when_not_computed(self):
        at = AdaptiveThreshold(min_window=5)
        assert at.threshold == float("inf")

    def test_compute_returns_inf_below_min_window(self):
        at = AdaptiveThreshold(min_window=5)
        for _ in range(4):
            at.add_score(1.0)
        assert at.compute() == float("inf")

    def test_mad_method(self):
        torch.manual_seed(0)
        at = AdaptiveThreshold(method="mad", multiplier=3.0, min_window=5)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for s in scores:
            at.add_score(s)
        th = at.compute()
        expected_median = 5.5
        mad = float(np.median(np.abs(np.array(scores) - expected_median)))
        expected_th = expected_median + 3.0 * mad
        assert th == pytest.approx(expected_th)

    def test_mad_handles_zero_mad(self):
        at = AdaptiveThreshold(method="mad", multiplier=3.0, min_window=5)
        for _ in range(10):
            at.add_score(5.0)
        th = at.compute()
        assert th == 5.0

    def test_add_score_with_labels(self):
        at = AdaptiveThreshold(method="validation_f1", min_window=5)
        for i in range(10):
            at.add_score(float(i), label=i % 2)
        assert at.compute() != float("inf")

    def test_validation_f1_finds_reasonable_threshold(self):
        at = AdaptiveThreshold(method="validation_f1", min_window=5)
        for i in range(10):
            at.add_score(float(i), label=0 if i < 5 else 1)
        th = at.compute()
        assert th != float("inf")

    def test_reset_clears_state(self):
        at = AdaptiveThreshold(min_window=5)
        for _ in range(10):
            at.add_score(1.0)
        at.compute()
        at.reset()
        assert at.threshold == float("inf")
        assert at.compute() == float("inf")

    def test_thread_safety(self):
        at = AdaptiveThreshold(method="mad", multiplier=3.0, min_window=5)
        errors: List[Exception] = []

        def adder():
            try:
                for _ in range(100):
                    at.add_score(float(np.random.rand()))
            except Exception as e:
                errors.append(e)

        def computer():
            try:
                for _ in range(100):
                    at.compute()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=computer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


class TestStreamingDetector:
    @pytest.fixture
    def detector(self):
        torch.manual_seed(0)
        model = MockGraphIDS()
        dyn_graph = DynamicGraph(edge_feature_dim=5)
        det = StreamingDetector(
            model=model,
            dynamic_graph=dyn_graph,
            window_size=4,
            embedding_dim=8,
            device="cpu",
            threshold_method="mad",
            threshold_multiplier=10.0,
            adaptive_threshold=True,
            adaptation_window=100,
        )
        return det

    def test_process_flow_returns_score_bool_tuple(self, detector):
        score, is_anom = detector.process_flow(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            edge_features=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamp=1000.0,
            label=0,
        )
        assert isinstance(score, float)
        assert isinstance(is_anom, bool)

    def test_process_flow_handles_empty_graph(self):
        torch.manual_seed(0)
        model = MockGraphIDS()
        dyn_graph = DynamicGraph(edge_feature_dim=5)
        det = StreamingDetector(
            model=model,
            dynamic_graph=dyn_graph,
            window_size=4,
            embedding_dim=8,
            device="cpu",
            threshold_method="mad",
            threshold_multiplier=10.0,
            adaptive_threshold=True,
            adaptation_window=100,
        )
        score, is_anom = det.process_flow(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            edge_features=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamp=1000.0,
            label=0,
        )
        assert isinstance(score, float)
        assert isinstance(is_anom, bool)

    def test_process_batch_returns_list_of_tuples(self, detector):
        df = pd.DataFrame({
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.2"],
            "IPV4_DST_ADDR": ["10.0.0.3", "10.0.0.4"],
            "FLOW_START_MILLISECONDS": [1000.0, 1001.0],
            "feat1": [1.0, 2.0],
            "feat2": [3.0, 4.0],
            "feat3": [5.0, 6.0],
            "feat4": [7.0, 8.0],
            "feat5": [9.0, 10.0],
            "Label": [0, 1],
        })
        results = detector.process_batch(df)
        assert len(results) == 2
        for score, is_anom in results:
            assert isinstance(score, float)
            assert isinstance(is_anom, bool)

    def test_get_statistics_returns_correct_keys(self, detector):
        stats = detector.get_statistics()
        expected_keys = {
            "total_flows", "total_alerts", "alert_rate",
            "current_threshold", "buffer_fill", "method",
            "adaptive_threshold",
        }
        assert set(stats.keys()) == expected_keys

    def test_update_threshold_returns_threshold(self, detector):
        for _ in range(20):
            detector.thresholder.add_score(float(np.random.rand()))
        th = detector.update_threshold(force=True)
        assert isinstance(th, float)

    def test_reset_clears_counters(self, detector):
        detector.process_flow(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            edge_features=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamp=1000.0,
        )
        detector.reset()
        stats = detector.get_statistics()
        assert stats["total_flows"] == 0
        assert stats["total_alerts"] == 0
        assert stats["buffer_fill"] == 0
        assert stats["current_threshold"] == float("inf")

    def test_multiple_flows_increase_counter(self, detector):
        for i in range(3):
            detector.process_flow(
                src_ip=f"10.0.0.{i}",
                dst_ip=f"10.0.0.{i + 1}",
                edge_features=[1.0, 2.0, 3.0, 4.0, 5.0],
                timestamp=1000.0 + i,
            )
        stats = detector.get_statistics()
        assert stats["total_flows"] == 3
