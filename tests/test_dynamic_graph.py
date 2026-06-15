import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

from utils.dynamic_graph import DynamicGraph


@pytest.fixture
def graph():
    return DynamicGraph(edge_feature_dim=3, feature_names=["f1", "f2", "f3"])


@pytest.fixture
def populated_graph(graph):
    graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0, label=0)
    graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 2000.0, label=1)
    graph.add_edge("10.0.0.1", "10.0.0.3", [0.7, 0.8, 0.9], 3000.0, label=0)
    return graph


class TestNodeMapping:
    def test_get_node_id_creates_new_id_for_unseen_ip(self, graph):
        nid = graph.get_node_id("10.0.0.1")
        assert nid == 0, "First IP should get ID 0"
        assert graph.num_nodes == 1

    def test_get_node_id_returns_same_id_for_same_ip(self, graph):
        nid1 = graph.get_node_id("10.0.0.1")
        nid2 = graph.get_node_id("10.0.0.1")
        assert nid1 == nid2

    def test_has_node_works_correctly(self, graph):
        graph.get_node_id("10.0.0.1")
        assert graph.has_node("10.0.0.1") is True
        assert graph.has_node("10.0.0.2") is False

    def test_max_nodes_limit_raises_runtime_error(self):
        small = DynamicGraph(edge_feature_dim=2, max_nodes=2)
        small.get_node_id("10.0.0.1")
        small.get_node_id("10.0.0.2")
        with pytest.raises(RuntimeError, match="Max nodes limit"):
            small.get_node_id("10.0.0.3")

    def test_inverse_node_map_is_consistent(self, graph):
        nid = graph.get_node_id("10.0.0.42")
        assert graph.inverse_node_map[nid] == "10.0.0.42"


class TestEdgeAddition:
    def test_add_edge_returns_sequential_edge_ids(self, graph):
        eid0 = graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0)
        eid1 = graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 2000.0)
        assert eid0 == 0
        assert eid1 == 1

    def test_add_edge_updates_node_count(self, graph):
        graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0)
        assert graph.num_nodes == 2
        graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 2000.0)
        assert graph.num_nodes == 3
        graph.add_edge("10.0.0.1", "10.0.0.3", [0.7, 0.8, 0.9], 3000.0)
        assert graph.num_nodes == 3

    def test_add_edge_updates_edge_count(self, graph):
        assert graph.num_edges == 0
        graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0)
        assert graph.num_edges == 1
        graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 2000.0)
        assert graph.num_edges == 2

    def test_add_edges_batch_processes_dataframe(self, graph):
        df = pd.DataFrame({
            "src": ["10.0.0.1", "10.0.0.2"],
            "dst": ["10.0.0.2", "10.0.0.3"],
            "f1": [0.1, 0.4],
            "f2": [0.2, 0.5],
            "f3": [0.3, 0.6],
            "ts": [1000.0, 2000.0],
            "Label": [0, 1],
        })
        edge_ids = graph.add_edges_batch(df, "ts", "src", "dst", ["f1", "f2", "f3"])
        np.testing.assert_array_equal(edge_ids, np.array([0, 1]))
        assert graph.num_edges == 2
        assert graph.num_nodes == 3

    def test_edge_attributes_are_stored_correctly(self, populated_graph):
        assert populated_graph._edge_attr[0] == [0.1, 0.2, 0.3]
        assert populated_graph._edge_attr[1] == [0.4, 0.5, 0.6]
        assert populated_graph._edge_attr[2] == [0.7, 0.8, 0.9]

    def test_edge_timestamps_are_stored_correctly(self, populated_graph):
        assert populated_graph._edge_timestamps[0] == 1000.0
        assert populated_graph._edge_timestamps[1] == 2000.0
        assert populated_graph._edge_timestamps[2] == 3000.0

    def test_edge_labels_are_stored_correctly(self, populated_graph):
        assert populated_graph._edge_labels[0] == 0
        assert populated_graph._edge_labels[1] == 1
        assert populated_graph._edge_labels[2] == 0


class TestEdgeRemoval:
    def test_remove_edge_marks_edge_inactive(self, populated_graph):
        assert populated_graph.num_edges == 3
        result = populated_graph.remove_edge(0)
        assert result is True
        assert populated_graph._edge_active[0] is False
        assert populated_graph.num_edges == 2

    def test_remove_edge_returns_false_for_invalid_id(self, populated_graph):
        assert populated_graph.remove_edge(999) is False

    def test_edge_count_decreases_after_removal(self, populated_graph):
        populated_graph.remove_edge(0)
        assert populated_graph.num_edges == 2
        populated_graph.remove_edge(1)
        assert populated_graph.num_edges == 1

    def test_remove_stale_edges_removes_old_edges(self, populated_graph):
        removed = populated_graph.remove_stale_edges(current_time=3500.0, max_age_ms=1000.0)
        assert removed == 2
        assert populated_graph.num_edges == 1

    def test_remove_stale_edges_keeps_recent_edges(self, populated_graph):
        removed = populated_graph.remove_stale_edges(current_time=3500.0, max_age_ms=1000.0)
        assert removed == 2
        assert populated_graph._edge_active[2] is True
        assert populated_graph.num_edges == 1

    def test_remove_stale_edges_returns_correct_count(self, populated_graph):
        result = populated_graph.remove_stale_edges(current_time=1500.0, max_age_ms=400.0)
        assert result == 1


class TestNodeRemoval:
    def test_remove_node_removes_all_incident_edges(self, populated_graph):
        node_id = populated_graph.get_node_id("10.0.0.1")
        result = populated_graph.remove_node(node_id)
        assert result is True
        assert populated_graph._edge_active[0] is False
        assert populated_graph._edge_active[1] is True
        assert populated_graph._edge_active[2] is False

    def test_remove_node_removes_from_node_map(self, populated_graph):
        node_id = populated_graph.get_node_id("10.0.0.1")
        populated_graph.remove_node(node_id)
        assert populated_graph.has_node("10.0.0.1") is False
        assert node_id not in populated_graph.inverse_node_map

    def test_remove_node_returns_false_for_unknown_node(self, populated_graph):
        assert populated_graph.remove_node(999) is False

    def test_remove_inactive_nodes_removes_stale_nodes(self, graph):
        graph.max_edge_age_ms = 1000.0
        graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0, label=0)
        graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 5000.0, label=1)
        removed = graph.remove_inactive_nodes(max_age_ms=1000.0, current_time=3000.0)
        assert removed == 1
        assert graph.has_node("10.0.0.1") is False
        assert graph.has_node("10.0.0.2") is True
        assert graph.has_node("10.0.0.3") is True

    def test_remove_inactive_nodes_keeps_active_nodes(self, graph):
        graph.max_edge_age_ms = 5000.0
        graph.add_edge("10.0.0.1", "10.0.0.2", [0.1, 0.2, 0.3], 1000.0, label=0)
        graph.add_edge("10.0.0.2", "10.0.0.3", [0.4, 0.5, 0.6], 2000.0, label=1)
        removed = graph.remove_inactive_nodes(max_age_ms=5000.0, current_time=3000.0)
        assert removed == 0
        assert graph.num_nodes == 3


class TestSnapshotGeneration:
    def test_get_current_graph_returns_data_with_correct_shape(self, populated_graph):
        data = populated_graph.get_current_graph()
        assert isinstance(data, Data)
        assert data.x.shape == (3, 3)
        assert data.edge_index.shape == (2, 3)
        assert data.edge_attr.shape == (3, 3)
        assert data.edge_labels.shape == (3,)
        assert data.num_nodes == 3

    def test_get_current_graph_handles_empty_graph(self, graph):
        data = graph.get_current_graph()
        assert isinstance(data, Data)
        assert data.x.shape == (0, 3)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 3)
        assert data.edge_labels.shape == (0,)
        assert data.num_nodes == 0

    def test_get_snapshot_filters_by_time_range(self, populated_graph):
        data = populated_graph.get_snapshot(1500.0, 2500.0)
        assert data.num_nodes == 2
        assert data.edge_index.shape[1] == 1

    def test_get_snapshot_handles_empty_time_range(self, populated_graph):
        data = populated_graph.get_snapshot(5000.0, 6000.0)
        assert isinstance(data, Data)
        assert data.num_nodes == 0
        assert data.edge_index.shape[1] == 0

    def test_node_features_are_all_ones(self, populated_graph):
        data = populated_graph.get_current_graph()
        assert torch.all(data.x == 1.0)

    def test_edge_features_match_input(self, populated_graph):
        data = populated_graph.get_current_graph()
        expected = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        assert torch.equal(data.edge_attr, expected)

    def test_get_snapshot_dataframe_returns_correct_columns(self, populated_graph):
        df = populated_graph.get_snapshot_dataframe(0.0, 5000.0)
        assert list(df.columns) == ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "f1", "f2", "f3"]
        assert len(df) == 3

    def test_get_snapshot_dataframe_empty_range(self, populated_graph):
        df = populated_graph.get_snapshot_dataframe(5000.0, 6000.0)
        assert isinstance(df, pd.DataFrame)
        assert df.empty is True


class TestQueryMethods:
    def test_num_nodes_matches_len_node_map(self, populated_graph):
        assert populated_graph.num_nodes == len(populated_graph.node_map)

    def test_num_edges_matches_active_edges_count(self, populated_graph):
        expected = sum(populated_graph._edge_active)
        assert populated_graph.num_edges == expected

    def test_get_node_ids_returns_all_ids(self, populated_graph):
        ids = populated_graph.get_node_ids()
        assert sorted(ids) == [0, 1, 2]

    def test_get_node_ids_empty(self, graph):
        assert graph.get_node_ids() == []

    def test_get_edge_ids_for_node_returns_incident_edges(self, populated_graph):
        nid = populated_graph.get_node_id("10.0.0.1")
        eids = populated_graph.get_edge_ids_for_node(nid)
        assert sorted(eids) == [0, 2]

    def test_get_node_degree_matches_edge_count_for_node(self, populated_graph):
        nid = populated_graph.get_node_id("10.0.0.1")
        degree = populated_graph.get_node_degree(nid)
        assert degree == 2

    def test_get_node_degree_after_removal(self, populated_graph):
        nid = populated_graph.get_node_id("10.0.0.1")
        populated_graph.remove_edge(0)
        assert populated_graph.get_node_degree(nid) == 1


class TestReset:
    def test_reset_clears_all_state(self, populated_graph):
        populated_graph.reset()
        assert populated_graph.num_nodes == 0
        assert populated_graph.num_edges == 0
        assert populated_graph.node_map == {}
        assert populated_graph.inverse_node_map == {}
        assert populated_graph.get_node_ids() == []

    def test_after_reset_num_nodes_and_num_edges_are_zero(self, populated_graph):
        populated_graph.reset()
        assert populated_graph.num_nodes == 0
        assert populated_graph.num_edges == 0


class TestScaler:
    def test_fit_scaler_creates_minmax_scaler(self, graph):
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "f3": [7.0, 8.0, 9.0]})
        graph.fit_scaler(df, ["f1", "f2", "f3"])
        assert isinstance(graph.scaler, MinMaxScaler)
        assert graph.scaler is not None

    def test_transform_features_transforms_correctly(self, graph):
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "f3": [7.0, 8.0, 9.0]})
        graph.fit_scaler(df, ["f1", "f2", "f3"])
        result = graph.transform_features(df, ["f1", "f2", "f3"])
        expected = pd.DataFrame(
            {"f1": [0.0, 0.5, 1.0], "f2": [0.0, 0.5, 1.0], "f3": [0.0, 0.5, 1.0]}
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_inverse_transform_features_inverts_correctly(self, graph):
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "f3": [7.0, 8.0, 9.0]})
        graph.fit_scaler(df, ["f1", "f2", "f3"])
        transformed = graph.transform_features(df, ["f1", "f2", "f3"])
        inverted = graph.inverse_transform_features(
            transformed[["f1", "f2", "f3"]].to_numpy()
        )
        np.testing.assert_array_almost_equal(inverted, df[["f1", "f2", "f3"]].to_numpy())

    def test_transform_before_fit_raises_error(self, graph):
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]})
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            graph.transform_features(df, ["f1", "f2", "f3"])

    def test_inverse_transform_before_fit_raises_error(self, graph):
        arr = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            graph.inverse_transform_features(arr)
