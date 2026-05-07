import torch
import torch.nn as nn


class EdgeAwareGraphSAGELayer(nn.Module):
    """
    共享 pair 子图的 edge-aware GraphSAGE layer。

    消息传递形式：
      gate_ji = sigmoid(MLP_e(edge_attr_ji))
      msg_ji = gate_ji * W h_j
      agg_i = mean_j msg_ji
      h_i_next = MLP([h_i || agg_i])
    """

    def __init__(self, input_dim: int, output_dim: int, edge_attr_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        num_nodes = node_features.size(0)
        if num_nodes == 0:
            return node_features.new_zeros((0, self.output_dim))

        residual = node_features if self.input_dim == self.output_dim else None
        aggregated_messages = node_features.new_zeros((num_nodes, self.output_dim))

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            projected_src = self.message_linear(node_features)
            edge_gates = torch.sigmoid(self.edge_gate_mlp(edge_attr))
            messages = projected_src[src] * edge_gates
            aggregated_messages.index_add_(0, dst, messages)

            node_degrees = node_features.new_zeros((num_nodes, 1))
            node_degrees.index_add_(0, dst, torch.ones((dst.size(0), 1), device=node_features.device, dtype=node_features.dtype))
            aggregated_messages = aggregated_messages / node_degrees.clamp_min(1.0)

        updated_node_features = self.update_mlp(torch.cat([node_features, aggregated_messages], dim=1))
        if residual is not None:
            updated_node_features = updated_node_features + residual

        updated_node_features = self.norm(updated_node_features)
        updated_node_features = self.activation(updated_node_features)
        return updated_node_features


class EdgeAwareGraphSAGE(nn.Module):
    """
    两层共享 pair 子图 GraphSAGE 编码器。
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, edge_attr_dim: int):
        super().__init__()
        self.layer1 = EdgeAwareGraphSAGELayer(input_dim=input_dim, output_dim=hidden_dim, edge_attr_dim=edge_attr_dim)
        self.layer2 = EdgeAwareGraphSAGELayer(input_dim=hidden_dim, output_dim=out_dim, edge_attr_dim=edge_attr_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        if node_features.size(0) == 0:
            return node_features.new_zeros((0, self.layer2.output_dim))
        hidden = self.layer1(node_features=node_features, edge_index=edge_index, edge_attr=edge_attr)
        hidden = self.layer2(node_features=hidden, edge_index=edge_index, edge_attr=edge_attr)
        return hidden


class SubgraphGCN(EdgeAwareGraphSAGE):
    """
    为兼容当前项目已有 import / 实例化方式，保留 SubgraphGCN 类名。
    实际内部实现已经切换为共享 pair 子图的 edge-aware GraphSAGE。
    """

    pass
