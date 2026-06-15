import torch
from torch_geometric.data import Data

from models.graphids import ColdStartNodeInitializer, GraphIDS, SAGELayer


class TestSAGELayer:
    def test_without_temporal_weights_backward_compat(self):
        torch.manual_seed(0)
        layer = SAGELayer(ndim_in=4, edim_in=3, edim_out=8, agg_type="mean", dropout_rate=0.0)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = layer(edge_index, edge_attr, edge_couples, num_nodes=3)
        assert out.shape == (2, 8)

    def test_with_temporal_weights_produces_same_shape_different_values(self):
        torch.manual_seed(0)
        layer = SAGELayer(ndim_in=4, edim_in=3, edim_out=8, agg_type="mean", dropout_rate=0.0)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        temporal_weights = torch.tensor([0.5, 2.0])
        out_no_tw = layer(edge_index, edge_attr, edge_couples, num_nodes=3)
        out_with_tw = layer(edge_index, edge_attr, edge_couples, num_nodes=3, temporal_weights=temporal_weights)
        assert out_with_tw.shape == (2, 8)
        assert not torch.allclose(out_no_tw, out_with_tw)

    def test_zero_temporal_weight_equals_zero_contribution(self):
        torch.manual_seed(0)
        layer = SAGELayer(ndim_in=4, edim_in=3, edim_out=8, agg_type="mean", dropout_rate=0.0)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out_ones = layer(edge_index, edge_attr, edge_couples, num_nodes=3, temporal_weights=torch.ones(2))
        out_zeros = layer(edge_index, edge_attr, edge_couples, num_nodes=3, temporal_weights=torch.zeros(2))
        assert out_zeros.shape == (2, 8)
        assert not torch.allclose(out_ones, out_zeros)

    def test_node_mask_filters_edges_correctly(self):
        torch.manual_seed(0)
        layer = SAGELayer(ndim_in=4, edim_in=3, edim_out=8, agg_type="mean", dropout_rate=0.0)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out_full = layer(edge_index, edge_attr, edge_couples, num_nodes=3)
        out_masked = layer(edge_index, edge_attr, edge_couples, num_nodes=3, node_mask=torch.tensor([0]))
        assert out_masked.shape == (2, 8)
        assert not torch.allclose(out_full, out_masked)

    def test_handles_nan_from_zero_incident_edge_nodes(self):
        torch.manual_seed(0)
        layer = SAGELayer(ndim_in=4, edim_in=3, edim_out=8, agg_type="mean", dropout_rate=0.0)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = layer(edge_index, edge_attr, edge_couples, num_nodes=4)
        assert out.shape == (2, 8)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()


class TestColdStartNodeInitializer:
    def test_neighbor_mean_fills_nan_embeddings(self):
        initializer = ColdStartNodeInitializer(ndim=4, strategy="neighbor_mean")
        node_embeddings = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [5.0, 6.0, 7.0, 8.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        result = initializer.initialize_nodes(
            torch.tensor([1]), node_embeddings, edge_index, num_nodes=3,
        )
        expected = torch.tensor([3.0, 4.0, 5.0, 6.0])
        assert torch.allclose(result[1], expected)

    def test_default_embedding_strategy(self):
        initializer = ColdStartNodeInitializer(ndim=4, strategy="default_embedding")
        assert initializer.default_embedding is not None
        node_embeddings = torch.full((2, 4), float("nan"))
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        result = initializer.initialize_nodes(
            torch.tensor([0]), node_embeddings, edge_index, num_nodes=2,
        )
        assert not torch.isnan(result[0]).any()
        assert torch.allclose(result[0], initializer.default_embedding)

    def test_leaves_non_nan_embeddings_unchanged(self):
        initializer = ColdStartNodeInitializer(ndim=4, strategy="neighbor_mean")
        node_embeddings = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
        ])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        result = initializer.initialize_nodes(
            torch.tensor([0, 1]), node_embeddings, edge_index, num_nodes=2,
        )
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


class TestGraphIDS:
    def test_encode_edges_produces_correct_shape(self):
        torch.manual_seed(0)
        model = GraphIDS(
            ndim_in=4, edim_in=3, edim_out=8, embed_dim=4,
            num_heads=2, num_layers=1, window_size=4,
            dropout=0.0, ae_dropout=0.0,
        )
        data = Data(
            x=torch.ones(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr=torch.rand(2, 3),
            num_nodes=3,
        )
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = model.encode_edges(data, edge_couples)
        assert out.shape == (2, 8)

    def test_forward_with_temporal_weights(self):
        torch.manual_seed(0)
        model = GraphIDS(
            ndim_in=4, edim_in=3, edim_out=8, embed_dim=4,
            num_heads=2, num_layers=1, window_size=4,
            dropout=0.0, ae_dropout=0.0,
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = model(
            edge_index, edge_attr, edge_couples, num_nodes=3,
            temporal_weights=torch.tensor([0.5, 2.0]),
        )
        assert out.shape == (2, 8)

    def test_forward_with_node_mask(self):
        torch.manual_seed(0)
        model = GraphIDS(
            ndim_in=4, edim_in=3, edim_out=8, embed_dim=4,
            num_heads=2, num_layers=1, window_size=4,
            dropout=0.0, ae_dropout=0.0,
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = model(
            edge_index, edge_attr, edge_couples, num_nodes=3,
            node_mask=torch.tensor([0]),
        )
        assert out.shape == (2, 8)

    def test_forward_without_new_params_backward_compat(self):
        torch.manual_seed(0)
        model = GraphIDS(
            ndim_in=4, edim_in=3, edim_out=8, embed_dim=4,
            num_heads=2, num_layers=1, window_size=4,
            dropout=0.0, ae_dropout=0.0,
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.rand(2, 3)
        edge_couples = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = model(edge_index, edge_attr, edge_couples, num_nodes=3)
        assert out.shape == (2, 8)
