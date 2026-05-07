import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from models.subgraph_gcn import SubgraphGCN
from utils.collate_gcn import build_pair_graph_pack, pair_graph_pack_to_torch, ROLE_FEAT_DIM, EDGE_ATTR_DIM
from utils.utils import NeighborSampler


class ContextEncoder(nn.Module):
    def __init__(self, seq_dim: int, stat_dim: int, q_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_dim + stat_dim + q_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, seq_emb: torch.Tensor, stat_emb: torch.Tensor, str_q: torch.Tensor):
        return self.net(torch.cat([seq_emb, stat_emb, str_q], dim=-1))


class NodeCCF(nn.Module):
    def __init__(self, seq_dim: int, str_dim: int, stat_dim: int, q_dim: int,
                 hidden_dim: int, out_dim: int, dropout: float = 0.1, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = float(res_scale)
        self.context_encoder = ContextEncoder(
            seq_dim=seq_dim,
            stat_dim=stat_dim,
            q_dim=q_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.conf_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, str_dim),
            nn.Sigmoid()
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, str_dim),
            nn.Tanh()
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(seq_dim + stat_dim + str_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, seq_emb: torch.Tensor, str_emb: torch.Tensor, stat_emb: torch.Tensor, str_q: torch.Tensor):
        context_emb = self.context_encoder(seq_emb=seq_emb, stat_emb=stat_emb, str_q=str_q)
        alpha = self.conf_mlp(context_emb)
        gate = self.gate_mlp(context_emb)
        residual = self.res_mlp(context_emb)
        gated_str = alpha * (gate * str_emb)
        recalibrated_str = gated_str + self.res_scale * residual
        node_emb = self.node_fuse(torch.cat([seq_emb, stat_emb, recalibrated_str], dim=-1))
        aux = {
            "context_emb": context_emb,
            "alpha": alpha,
            "gate": gate,
            "residual": residual,
            "gated_str": gated_str,
            "recalibrated_str": recalibrated_str,
        }
        return node_emb, aux


class NodeBaselineFusion(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim)
        )

    def forward(self, seq_emb: torch.Tensor, str_emb: torch.Tensor, stat_emb: torch.Tensor):
        node_emb = self.fuse(torch.cat([seq_emb, str_emb, stat_emb], dim=-1))
        zero_scalar = torch.zeros(seq_emb.size(0), 1, device=seq_emb.device, dtype=seq_emb.dtype)
        zero_tensor = torch.zeros_like(seq_emb)
        return node_emb, {
            "context_emb": zero_tensor,
            "alpha": zero_scalar,
            "gate": zero_tensor,
            "residual": zero_tensor,
            "gated_str": zero_tensor,
            "recalibrated_str": str_emb,
        }


class PairMemoryEncoder(nn.Module):
    def __init__(self, pair_raw_dim: int, pair_time_raw_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.direct_dim = 8
        self.struct_dim = pair_raw_dim - self.direct_dim
        self.time_dim = pair_time_raw_dim
        self.direct_encoder = nn.Sequential(
            nn.Linear(self.direct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.struct_encoder = nn.Sequential(
            nn.Linear(self.struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        self.memory_fuse = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, pair_raw: torch.Tensor, pair_time_raw: torch.Tensor):
        direct_raw = pair_raw[:, :self.direct_dim]
        struct_raw = pair_raw[:, self.direct_dim:]
        direct_emb = self.direct_encoder(direct_raw)
        struct_emb = self.struct_encoder(struct_raw)
        time_emb = self.time_encoder(pair_time_raw)
        gate_inputs = torch.cat([direct_emb, struct_emb, time_emb], dim=-1)
        gates = self.memory_gate(gate_inputs)
        gated_direct = gates[:, 0:1] * direct_emb
        gated_struct = gates[:, 1:2] * struct_emb
        gated_time = gates[:, 2:3] * time_emb
        pair_memory = self.memory_fuse(torch.cat([gated_direct, gated_struct, gated_time], dim=-1))
        aux = {
            "pair_memory": pair_memory,
            "memory_direct_gate": gates[:, 0:1],
            "memory_struct_gate": gates[:, 1:2],
            "memory_time_gate": gates[:, 2:3],
        }
        return pair_memory, aux


class PairCCF(nn.Module):
    def __init__(self, node_dim: int, pair_memory_dim: int, hidden_dim: int, dropout: float = 0.1,
                 pair_res_scale: float = 0.02, memory_bias_scale: float = 0.1,
                 pair_ccf_variant: str = 'adaptive',
                 pair_gamma_mode: str = 'adaptive',
                 pair_beta_mode: str = 'adaptive',
                 pair_residual_mode: str = 'adaptive'):
        super().__init__()
        if pair_ccf_variant not in ['adaptive', 'no_pair_ccf', 'legacy']:
            raise ValueError(f'Unsupported pair_ccf_variant {pair_ccf_variant}!')
        self.pair_ccf_variant = pair_ccf_variant
        self.pair_res_scale = float(pair_res_scale)
        self.memory_bias_scale = float(memory_bias_scale)
        self.pair_gamma_mode = pair_gamma_mode
        self.pair_beta_mode = pair_beta_mode
        self.pair_residual_mode = pair_residual_mode
        pair_base_dim = node_dim * 4
        pair_context_dim = pair_base_dim + pair_memory_dim
        self.pair_gate = nn.Sequential(
            nn.Linear(pair_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_base_dim),
            nn.Sigmoid()
        )
        self.pair_res = nn.Sequential(
            nn.Linear(pair_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_base_dim),
            nn.Tanh()
        )
        self.pair_conf = nn.Sequential(
            nn.Linear(pair_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.memory_bias = nn.Sequential(
            nn.Linear(pair_memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.pred_head = nn.Sequential(
            nn.Linear(pair_base_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, src_node: torch.Tensor, dst_node: torch.Tensor, pair_memory: torch.Tensor):
        pair_base = torch.cat([
            src_node,
            dst_node,
            src_node * dst_node,
            torch.abs(src_node - dst_node)
        ], dim=-1)
        pair_context = torch.cat([pair_base, pair_memory], dim=-1)
        beta = self.pair_gate(pair_context)
        if self.pair_beta_mode == 'ones':
            beta = torch.ones_like(beta)
        pair_res = self.pair_res(pair_context)
        if self.pair_residual_mode == 'zero':
            pair_res = torch.zeros_like(pair_res)
        pair_ccf = beta * pair_base + self.pair_res_scale * pair_res
        if self.pair_ccf_variant == 'no_pair_ccf':
            gamma = torch.zeros(pair_base.size(0), 1, device=pair_base.device, dtype=pair_base.dtype)
            pair_emb = pair_base
        elif self.pair_ccf_variant == 'legacy':
            gamma = torch.ones(pair_base.size(0), 1, device=pair_base.device, dtype=pair_base.dtype)
            pair_emb = pair_ccf
        else:
            gamma = self.pair_conf(pair_context)
            if self.pair_gamma_mode == 'ones':
                gamma = torch.ones_like(gamma)
            elif self.pair_gamma_mode == 'zeros':
                gamma = torch.zeros_like(gamma)
            pair_emb = (1.0 - gamma) * pair_base + gamma * pair_ccf
        memory_bias = self.memory_bias(pair_memory)
        logit = self.pred_head(pair_emb) + self.memory_bias_scale * memory_bias
        aux = {
            "beta": beta,
            "gamma": gamma,
            "pair_res": pair_res,
            "pair_base": pair_base,
            "pair_ccf": pair_ccf,
            "pair_emb": pair_emb,
            "pair_memory": pair_memory,
            "memory_bias": memory_bias,
        }
        return logit, aux


class PairFusionHead(nn.Module):
    def __init__(self, node_dim: int, pair_memory_dim: int, hidden_dim: int, dropout: float = 0.1, mode: str = "mlp"):
        super().__init__()
        self.mode = mode
        pair_dim = node_dim * 4 + pair_memory_dim
        if mode == "concat":
            self.pred = nn.Linear(pair_dim, 1)
        elif mode == "linear":
            self.pred = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        elif mode == "mlp":
            self.pred = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unsupported pair fusion mode {mode}!")

    def forward(self, src_node: torch.Tensor, dst_node: torch.Tensor, pair_memory: torch.Tensor):
        pair_base = torch.cat([src_node, dst_node, src_node * dst_node, torch.abs(src_node - dst_node)], dim=-1)
        pair_inputs = torch.cat([pair_base, pair_memory], dim=-1)
        logit = self.pred(pair_inputs)
        zero_scalar = torch.zeros(pair_base.size(0), 1, device=pair_base.device, dtype=pair_base.dtype)
        zero_tensor = torch.zeros_like(pair_base)
        aux = {
            "beta": torch.ones_like(pair_base),
            "gamma": zero_scalar,
            "pair_res": zero_tensor,
            "pair_base": pair_base,
            "pair_ccf": pair_base,
            "pair_emb": pair_base,
            "pair_memory": pair_memory,
            "memory_bias": zero_scalar,
        }
        return logit, aux


class S3Former(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, recent_k: int = 16, max_neighbors_gcn: int = 20,
                 common_neighbors_topk: int = 8, pair_graph_score_alpha: float = 1.0, pair_graph_score_beta: float = 1e-6,
                 pair_graph_score_gamma: float = 1.0, pair_graph_common_lambda: float = 0.5,
                 short_window_size: int = 8, use_short_term: bool = True, short_hidden_dim: int = 128,
                 short_num_layers: int = 2, short_conv_kernel_size: int = 3, short_dropout: float = 0.1,
                 short_pooling: str = 'mean', short_fusion_mode: str = 'gate',
                 pair_ccf_variant: str = 'adaptive', device: str = 'cpu'):
        super(S3Former, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.recent_k = int(recent_k)
        self.max_neighbors_gcn = int(max_neighbors_gcn)
        self.common_neighbors_topk = int(common_neighbors_topk)
        self.pair_graph_score_alpha = float(pair_graph_score_alpha)
        self.pair_graph_score_beta = float(pair_graph_score_beta)
        self.pair_graph_score_gamma = float(pair_graph_score_gamma)
        self.pair_graph_common_lambda = float(pair_graph_common_lambda)
        self.pair_ccf_variant = pair_ccf_variant
        self.role_feat_dim = ROLE_FEAT_DIM
        self.edge_attr_dim = EDGE_ATTR_DIM
        self.disable_struct = False
        self.disable_stat = False
        self.disable_short_term = False
        self.use_seq_channel = True
        self.use_struct_channel = True
        self.use_stat_channel = True
        self.node_fusion_mode = "ccf"
        self.pair_fusion_mode = "ccf"
        self.pair_gamma_mode = "adaptive"
        self.pair_beta_mode = "adaptive"
        self.pair_residual_mode = "adaptive"
        self.pair_memory_mode = "learned"
        self.pair_raw_structural_zero = False
        self.stat_source_mode = "online"
        self.stat_decay_mode = "fixed"
        self.stat_feature_variant = "full"
        self.anchor_selection_mode = "recent_k"
        self.edge_construction_mode = "sparse_induced"
        self.common_neighbor_mode = "topk"

        self.short_window_size = int(short_window_size)
        self.use_short_term = bool(use_short_term) and not self.disable_short_term
        self.short_hidden_dim = int(short_hidden_dim)
        self.short_num_layers = int(short_num_layers)
        self.short_conv_kernel_size = int(short_conv_kernel_size)
        self.short_dropout = float(short_dropout)
        self.short_pooling = short_pooling
        self.short_fusion_mode = short_fusion_mode

        self.gcn_feat_dim = self.channel_embedding_dim
        self.subgraph_gcn = SubgraphGCN(
            input_dim=self.node_feat_dim + self.role_feat_dim,
            hidden_dim=self.channel_embedding_dim,
            out_dim=self.gcn_feat_dim,
            edge_attr_dim=self.edge_attr_dim
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(self.patch_size * self.node_feat_dim, self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(self.patch_size * self.edge_feat_dim, self.channel_embedding_dim, bias=True),
            'time': nn.Linear(self.patch_size * self.time_feat_dim, self.channel_embedding_dim, bias=True),
            'gcn': nn.Linear(self.patch_size * self.gcn_feat_dim, self.channel_embedding_dim, bias=True),
        })

        self.sequence_num_channels = 3
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                attention_dim=self.sequence_num_channels * self.channel_embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        self.sequence_output_layer = nn.Linear(
            in_features=self.sequence_num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True
        )

        self.raw_stat_dim = 4
        self.pair_raw_stat_dim = 16
        self.pair_time_raw_dim = 8
        self.str_q_dim = 4
        self.structure_gcn_output_layer = nn.Linear(self.channel_embedding_dim, self.node_feat_dim, bias=False)
        self.structure_gcn_norm = nn.LayerNorm(self.node_feat_dim)
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.raw_stat_dim, self.node_feat_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.node_feat_dim, self.node_feat_dim),
            nn.LayerNorm(self.node_feat_dim)
        )
        self.pair_memory_encoder = PairMemoryEncoder(
            pair_raw_dim=self.pair_raw_stat_dim,
            pair_time_raw_dim=self.pair_time_raw_dim,
            hidden_dim=self.node_feat_dim,
            out_dim=self.node_feat_dim,
            dropout=self.dropout
        )
        self.stat_decay_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.node_ccf = NodeCCF(
            seq_dim=self.node_feat_dim,
            str_dim=self.node_feat_dim,
            stat_dim=self.node_feat_dim,
            q_dim=self.str_q_dim,
            hidden_dim=self.node_feat_dim,
            out_dim=self.node_feat_dim,
            dropout=self.dropout,
            res_scale=0.1
        )
        self.node_baseline_fuse = NodeBaselineFusion(
            node_dim=self.node_feat_dim,
            hidden_dim=self.node_feat_dim,
            dropout=self.dropout
        )
        self.pair_ccf = PairCCF(
            node_dim=self.node_feat_dim,
            hidden_dim=self.node_feat_dim,
            pair_memory_dim=self.node_feat_dim,
            dropout=self.dropout,
            pair_res_scale=0.02,
            memory_bias_scale=0.1,
            pair_ccf_variant=self.pair_ccf_variant,
            pair_gamma_mode=self.pair_gamma_mode,
            pair_beta_mode=self.pair_beta_mode,
            pair_residual_mode=self.pair_residual_mode
        )
        self.pair_plain_fusion = PairFusionHead(
            node_dim=self.node_feat_dim,
            pair_memory_dim=self.node_feat_dim,
            hidden_dim=self.node_feat_dim,
            dropout=self.dropout,
            mode="mlp" if self.pair_fusion_mode == "ccf" else self.pair_fusion_mode
        )

        if self.use_short_term:
            assert self.short_conv_kernel_size > 0, 'short_conv_kernel_size should be positive!'
            assert self.short_pooling in ['mean', 'attn'], f'Unsupported short_pooling {self.short_pooling}!'
            assert self.short_fusion_mode in ['gate'], f'Unsupported short_fusion_mode {self.short_fusion_mode}!'
            short_input_dim = self.node_feat_dim + self.edge_feat_dim + self.time_feat_dim
            self.short_input_projection = nn.Linear(short_input_dim, self.short_hidden_dim, bias=True)
            self.short_conv_layers = nn.ModuleList([
                nn.Conv1d(self.short_hidden_dim, self.short_hidden_dim, self.short_conv_kernel_size, bias=True)
                for _ in range(self.short_num_layers)
            ])
            self.short_layer_norms = nn.ModuleList([nn.LayerNorm(self.short_hidden_dim) for _ in range(self.short_num_layers)])
            self.short_output_layer = nn.Linear(self.short_hidden_dim, self.node_feat_dim, bias=True)
            self.short_attention_pooling = nn.Linear(self.short_hidden_dim, 1, bias=True)
            self.short_fusion_gate_layer = nn.Linear(self.node_feat_dim * 2, self.node_feat_dim, bias=True)
            self.short_conf_layer = nn.Sequential(
                nn.Linear(self.node_feat_dim * 2, self.node_feat_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.node_feat_dim, 1),
                nn.Sigmoid()
            )
            self.short_dropout_layer = nn.Dropout(self.short_dropout)

    def get_node_input_features(self, node_ids):
        if not isinstance(node_ids, torch.Tensor):
            node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
        else:
            node_ids = node_ids.to(self.device, dtype=torch.long, non_blocking=True)
        return self.node_raw_features[node_ids]

    def build_patch_mask_from_padded_ids(self, padded_nodes_neighbor_ids: np.ndarray):
        padded_ids_tensor = torch.from_numpy(padded_nodes_neighbor_ids).long().to(self.device)
        batch_size, seq_len = padded_ids_tensor.shape
        assert seq_len % self.patch_size == 0
        num_patches = seq_len // self.patch_size
        return (padded_ids_tensor.reshape(batch_size, num_patches, self.patch_size) != 0).any(dim=-1).float()

    def build_patch_mask_from_token_map(self, token_map: torch.Tensor):
        """
        根据共享 pair 子图的 token_map 构造结构通道自己的 patch mask。
        只有至少有一个 token 映射进 pair 子图的 patch 才算有效，避免零结构特征被错误纳入 pooling 分母。
        """
        batch_size, seq_len = token_map.shape
        assert seq_len % self.patch_size == 0
        num_patches = seq_len // self.patch_size
        if token_map.dtype == torch.bool:
            valid_tokens = token_map
        else:
            valid_tokens = token_map >= 0
        return valid_tokens.reshape(batch_size, num_patches, self.patch_size).any(dim=-1).float()

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor):
        mask = mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(dim=-1)).sum(dim=1) / denom

    def apply_patch_mask(self, x: torch.Tensor, patch_mask: torch.Tensor):
        return x * patch_mask.unsqueeze(dim=-1)

    def get_recent_window(self, padded_tensor: torch.Tensor, padded_nodes_neighbor_ids: np.ndarray):
        """
        从 padded 历史中截取最近 K 条有效历史交互。
        只看 [:, 1:] 历史部分，不包含第 0 位中心节点 token。
        返回:
          recent_window: (B, K, D)
          recent_mask: (B, K)
        """
        batch_size, seq_len, feat_dim = padded_tensor.shape
        window_size = max(self.short_window_size, 0)
        recent_window = torch.zeros(batch_size, window_size, feat_dim, device=padded_tensor.device, dtype=padded_tensor.dtype)
        recent_mask = torch.zeros(batch_size, window_size, device=padded_tensor.device, dtype=torch.float32)
        if window_size == 0 or seq_len <= 1:
            return recent_window, recent_mask
        history_neighbor_ids = padded_nodes_neighbor_ids[:, 1:]
        for batch_idx in range(batch_size):
            valid_history_indices = np.flatnonzero(history_neighbor_ids[batch_idx] != 0)
            if len(valid_history_indices) == 0:
                continue
            valid_history_indices = valid_history_indices[-window_size:] + 1
            valid_count = len(valid_history_indices)
            start_idx = window_size - valid_count
            recent_window[batch_idx, start_idx:] = padded_tensor[batch_idx, valid_history_indices, :]
            recent_mask[batch_idx, start_idx:] = 1.0
        return recent_window, recent_mask

    def masked_attention_pool(self, x: torch.Tensor, mask: torch.Tensor):
        batch_size, _, hidden_dim = x.shape
        pooled_x = torch.zeros(batch_size, hidden_dim, device=x.device, dtype=x.dtype)
        valid_mask = mask > 0
        if not valid_mask.any():
            return pooled_x
        attn_scores = self.short_attention_pooling(x).squeeze(dim=-1)
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        valid_rows = valid_mask.any(dim=1)
        if valid_rows.any():
            attn_weights = torch.softmax(attn_scores[valid_rows], dim=1)
            pooled_x[valid_rows] = torch.sum(x[valid_rows] * attn_weights.unsqueeze(dim=-1), dim=1)
        return pooled_x

    def compute_short_term_embeddings(self, padded_nodes_neighbor_ids: np.ndarray,
                                      padded_nodes_neighbor_node_raw_features: torch.Tensor,
                                      padded_nodes_edge_raw_features: torch.Tensor,
                                      padded_nodes_neighbor_time_features: torch.Tensor):
        """
        short-term 输入:
          node / edge / time 三种特征，最终拼成 (B, K, node_feat_dim + edge_feat_dim + time_feat_dim)
        输出:
          short_term_embeddings: (B, node_feat_dim)
        """
        batch_size = padded_nodes_neighbor_node_raw_features.shape[0]
        if not self.use_short_term or self.short_window_size <= 0:
            return torch.zeros(batch_size, self.node_feat_dim, device=padded_nodes_neighbor_node_raw_features.device)

        recent_node_features, recent_mask = self.get_recent_window(
            padded_tensor=padded_nodes_neighbor_node_raw_features,
            padded_nodes_neighbor_ids=padded_nodes_neighbor_ids
        )
        recent_edge_features, _ = self.get_recent_window(
            padded_tensor=padded_nodes_edge_raw_features,
            padded_nodes_neighbor_ids=padded_nodes_neighbor_ids
        )
        recent_time_features, _ = self.get_recent_window(
            padded_tensor=padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_ids=padded_nodes_neighbor_ids
        )

        short_term_tokens = torch.cat([recent_node_features, recent_edge_features, recent_time_features], dim=2)
        short_hidden_states = self.short_input_projection(short_term_tokens)
        short_hidden_states = short_hidden_states * recent_mask.unsqueeze(dim=-1)

        # causal conv: 先左侧 padding，再 Conv1d，保证当前位置只依赖当前及更早历史
        for short_conv_layer, short_layer_norm in zip(self.short_conv_layers, self.short_layer_norms):
            residual = short_hidden_states
            conv_inputs = short_hidden_states.transpose(1, 2)
            conv_inputs = F.pad(conv_inputs, (self.short_conv_kernel_size - 1, 0))
            conv_outputs = short_conv_layer(conv_inputs).transpose(1, 2)
            conv_outputs = F.gelu(conv_outputs)
            conv_outputs = self.short_dropout_layer(conv_outputs)
            short_hidden_states = short_layer_norm(residual + conv_outputs)
            short_hidden_states = short_hidden_states * recent_mask.unsqueeze(dim=-1)

        if self.short_pooling == 'mean':
            pooled_short_hidden_states = self.masked_mean(short_hidden_states, recent_mask)
        elif self.short_pooling == 'attn':
            pooled_short_hidden_states = self.masked_attention_pool(short_hidden_states, recent_mask)
        else:
            raise ValueError(f'Not implemented short_pooling {self.short_pooling}!')
        has_history = recent_mask.any(dim=1, keepdim=True).float()
        return self.short_output_layer(pooled_short_hidden_states) * has_history

    def fuse_long_short(self, long_term_embeddings: torch.Tensor, short_term_embeddings: torch.Tensor,
                        return_aux: bool = False):
        batch_size = long_term_embeddings.size(0)
        zero_eta = torch.zeros(batch_size, 1, device=long_term_embeddings.device, dtype=long_term_embeddings.dtype)
        zero_gate = torch.zeros_like(long_term_embeddings)
        zero_contribution = torch.zeros_like(long_term_embeddings)
        if not self.use_short_term or self.short_window_size <= 0:
            aux = {
                "short_eta": zero_eta,
                "short_gate": zero_gate,
                "long_norm": long_term_embeddings.norm(p=2, dim=1, keepdim=True),
                "short_norm": short_term_embeddings.norm(p=2, dim=1, keepdim=True),
                "short_contribution_norm": zero_contribution.norm(p=2, dim=1, keepdim=True),
            }
            return (long_term_embeddings, aux) if return_aux else long_term_embeddings
        if self.short_fusion_mode != 'gate':
            raise ValueError(f'Not implemented short_fusion_mode {self.short_fusion_mode}!')
        gate_inputs = torch.cat([long_term_embeddings, short_term_embeddings], dim=1)
        fusion_gate = torch.sigmoid(self.short_fusion_gate_layer(gate_inputs))
        short_eta = self.short_conf_layer(gate_inputs)
        short_contribution = short_eta * fusion_gate * short_term_embeddings
        seq_embeddings = long_term_embeddings + short_contribution
        aux = {
            "short_eta": short_eta,
            "short_gate": fusion_gate,
            "long_norm": long_term_embeddings.norm(p=2, dim=1, keepdim=True),
            "short_norm": short_term_embeddings.norm(p=2, dim=1, keepdim=True),
            "short_contribution_norm": short_contribution.norm(p=2, dim=1, keepdim=True),
        }
        return (seq_embeddings, aux) if return_aux else seq_embeddings

    def gcn_features_from_graph(self, pair_gcn_pack: dict):
        """
        读取共享 pair 子图 pack，并回填两份序列对齐的结构特征：
          src_padded_nodes_graph_features: (B, L_src, gcn_feat_dim)
          dst_padded_nodes_graph_features: (B, L_dst, gcn_feat_dim)
        同时返回 token 级结构有效 mask，供后续构造结构 patch mask。
        """
        if "pair" in pair_gcn_pack:
            pair_gcn_pack = pair_gcn_pack["pair"]

        uniq_nodes = pair_gcn_pack["uniq_nodes"].to(self.device, non_blocking=True)
        edge_index = pair_gcn_pack["edge_index"].to(self.device, non_blocking=True)
        edge_attr = pair_gcn_pack["edge_attr"].to(self.device, non_blocking=True)
        node_role_features = pair_gcn_pack["node_role_features"].to(self.device, non_blocking=True)
        src_token_map = pair_gcn_pack["src_token_map"].to(self.device, non_blocking=True)
        dst_token_map = pair_gcn_pack["dst_token_map"].to(self.device, non_blocking=True)

        src_batch_size, src_seq_len = src_token_map.shape
        dst_batch_size, dst_seq_len = dst_token_map.shape
        src_out = torch.zeros(src_batch_size, src_seq_len, self.gcn_feat_dim, device=self.device)
        dst_out = torch.zeros(dst_batch_size, dst_seq_len, self.gcn_feat_dim, device=self.device)

        if uniq_nodes.numel() == 0:
            return src_out, dst_out, src_token_map >= 0, dst_token_map >= 0

        node_input_features = torch.cat([self.get_node_input_features(uniq_nodes), node_role_features], dim=1)
        graph_outputs = self.subgraph_gcn(
            node_features=node_input_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        src_valid_mask = src_token_map >= 0
        dst_valid_mask = dst_token_map >= 0
        src_out[src_valid_mask] = graph_outputs[src_token_map[src_valid_mask]]
        dst_out[dst_valid_mask] = graph_outputs[dst_token_map[dst_valid_mask]]
        return src_out, dst_out, src_valid_mask, dst_valid_mask

    def compute_structure_embeddings(self, gcn_patch_features: torch.Tensor, gcn_patch_mask: torch.Tensor):
        gcn_pooled = self.masked_mean(gcn_patch_features, gcn_patch_mask)
        return self.structure_gcn_norm(self.structure_gcn_output_layer(gcn_pooled))

    def compute_struct_quality(self, pair_gcn_pack: dict, batch_size: int):
        if "pair" in pair_gcn_pack:
            pair_gcn_pack = pair_gcn_pack["pair"]

        def _metadata_tensor(name: str):
            value = pair_gcn_pack.get(name)
            if value is None:
                return torch.zeros(batch_size, device=self.device)
            return value.to(self.device, non_blocking=True).float()

        common_neighbor_counts = _metadata_tensor("common_neighbor_counts")
        densities = _metadata_tensor("densities")
        triangle_closure_scores = _metadata_tensor("triangle_closure_scores")
        src_valid_anchor_ratios = _metadata_tensor("src_valid_anchor_ratios")
        dst_valid_anchor_ratios = _metadata_tensor("dst_valid_anchor_ratios")

        common_quality = torch.stack([
            torch.log1p(common_neighbor_counts),
            densities,
            triangle_closure_scores,
        ], dim=-1)
        src_str_q = torch.cat([common_quality, src_valid_anchor_ratios.unsqueeze(dim=-1)], dim=-1)
        dst_str_q = torch.cat([common_quality, dst_valid_anchor_ratios.unsqueeze(dim=-1)], dim=-1)
        return src_str_q, dst_str_q

    def apply_channel_masks(self, seq_emb: torch.Tensor, str_emb: torch.Tensor, stat_emb: torch.Tensor, str_q: torch.Tensor):
        if self.disable_struct or not self.use_struct_channel:
            str_emb = torch.zeros_like(str_emb)
            str_q = torch.zeros_like(str_q)
        if self.disable_stat or not self.use_stat_channel:
            stat_emb = torch.zeros_like(stat_emb)
        if not self.use_seq_channel:
            seq_emb = torch.zeros_like(seq_emb)
        return seq_emb, str_emb, stat_emb, str_q

    def apply_stat_decay_scale(self, src_raw: torch.Tensor, dst_raw: torch.Tensor, pair_time_raw: torch.Tensor):
        if self.stat_decay_mode != "learnable":
            return src_raw, dst_raw, pair_time_raw
        scale = torch.sigmoid(self.stat_decay_scale)
        src_raw = src_raw.clone()
        dst_raw = dst_raw.clone()
        pair_time_raw = pair_time_raw.clone()
        src_raw[:, 2:4] = src_raw[:, 2:4] * scale
        dst_raw[:, 2:4] = dst_raw[:, 2:4] * scale
        pair_time_raw[:, :4] = pair_time_raw[:, :4] * scale
        return src_raw, dst_raw, pair_time_raw

    def compute_stat_embeddings(self, stat_inputs: dict, batch_size: int):
        if stat_inputs is None:
            src_raw = torch.zeros(batch_size, self.raw_stat_dim, device=self.device)
            dst_raw = torch.zeros(batch_size, self.raw_stat_dim, device=self.device)
            pair_raw = torch.zeros(batch_size, self.pair_raw_stat_dim, device=self.device)
            pair_time_raw = torch.zeros(batch_size, self.pair_time_raw_dim, device=self.device)
        else:
            src_raw = stat_inputs["src_raw"].to(self.device, non_blocking=True).float()
            dst_raw = stat_inputs["dst_raw"].to(self.device, non_blocking=True).float()
            pair_raw = stat_inputs["pair_raw"].to(self.device, non_blocking=True).float()
            pair_time_raw = stat_inputs["pair_time_raw"].to(self.device, non_blocking=True).float()

        if self.pair_raw_structural_zero:
            pair_raw = pair_raw.clone()
            pair_raw[:, 8:] = 0.0
        if self.disable_stat or self.pair_memory_mode == "zero":
            src_raw = torch.zeros_like(src_raw)
            dst_raw = torch.zeros_like(dst_raw)
            pair_raw = torch.zeros_like(pair_raw)
            pair_time_raw = torch.zeros_like(pair_time_raw)
        src_raw, dst_raw, pair_time_raw = self.apply_stat_decay_scale(src_raw, dst_raw, pair_time_raw)

        src_stat = self.stat_encoder(src_raw)
        dst_stat = self.stat_encoder(dst_raw)
        if self.disable_stat or self.pair_memory_mode == "zero":
            pair_memory = torch.zeros(batch_size, self.node_feat_dim, device=self.device, dtype=src_raw.dtype)
            zero_gate = torch.zeros(batch_size, 1, device=self.device, dtype=src_raw.dtype)
            memory_aux = {
                "pair_memory": pair_memory,
                "memory_direct_gate": zero_gate,
                "memory_struct_gate": zero_gate,
                "memory_time_gate": zero_gate,
            }
        else:
            pair_memory, memory_aux = self.pair_memory_encoder(pair_raw=pair_raw, pair_time_raw=pair_time_raw)
        raw_aux = {
            "src_raw": src_raw,
            "dst_raw": dst_raw,
            "pair_raw": pair_raw,
            "pair_time_raw": pair_time_raw,
            **memory_aux,
        }
        return src_stat, dst_stat, pair_memory, raw_aux

    def _compute_temporal_embeddings_from_padded_inputs(self,
                                                        src_padded_nodes_neighbor_ids: np.ndarray,
                                                        src_padded_nodes_edge_ids: np.ndarray,
                                                        src_padded_nodes_neighbor_times: np.ndarray,
                                                        dst_padded_nodes_neighbor_ids: np.ndarray,
                                                        dst_padded_nodes_edge_ids: np.ndarray,
                                                        dst_padded_nodes_neighbor_times: np.ndarray,
                                                        node_interact_times: np.ndarray,
                                                        pair_gcn_pack: dict,
                                                        stat_inputs: dict = None,
                                                        compute_pair: bool = True):
        """
        统一两条路径:
          1. 先跑当前版本已有的 long-term / structure fused 逻辑；
          2. 再追加 short-term branch 与 gate fusion。
        """
        src_patch_mask = self.build_patch_mask_from_padded_ids(src_padded_nodes_neighbor_ids)
        dst_patch_mask = self.build_patch_mask_from_padded_ids(dst_padded_nodes_neighbor_ids)

        if pair_gcn_pack is None:
            raise ValueError('pair_gcn_pack must not be None. Formal train/eval and old API should all use shared pair graph.')
        src_padded_nodes_gcn_features, dst_padded_nodes_gcn_features, src_gcn_token_mask, dst_gcn_token_mask = \
            self.gcn_features_from_graph(pair_gcn_pack)
        src_gcn_patch_mask = self.build_patch_mask_from_token_map(src_gcn_token_mask)
        dst_gcn_patch_mask = self.build_patch_mask_from_token_map(dst_gcn_token_mask)

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder
            )

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, src_patches_nodes_neighbor_time_features, \
        src_patches_nodes_gcn_features = self.get_patches(
            padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
            padded_nodes_gcn_features=src_padded_nodes_gcn_features,
            patch_size=self.patch_size
        )
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, dst_patches_nodes_neighbor_time_features, \
        dst_patches_nodes_gcn_features = self.get_patches(
            padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
            padded_nodes_gcn_features=dst_padded_nodes_gcn_features,
            patch_size=self.patch_size
        )

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_gcn_features = self.projection_layer['gcn'](src_patches_nodes_gcn_features)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_gcn_features = self.projection_layer['gcn'](dst_patches_nodes_gcn_features)

        src_patches_nodes_neighbor_node_raw_features = self.apply_patch_mask(src_patches_nodes_neighbor_node_raw_features, src_patch_mask)
        src_patches_nodes_edge_raw_features = self.apply_patch_mask(src_patches_nodes_edge_raw_features, src_patch_mask)
        src_patches_nodes_neighbor_time_features = self.apply_patch_mask(src_patches_nodes_neighbor_time_features, src_patch_mask)
        src_patches_nodes_gcn_features = self.apply_patch_mask(src_patches_nodes_gcn_features, src_gcn_patch_mask)
        dst_patches_nodes_neighbor_node_raw_features = self.apply_patch_mask(dst_patches_nodes_neighbor_node_raw_features, dst_patch_mask)
        dst_patches_nodes_edge_raw_features = self.apply_patch_mask(dst_patches_nodes_edge_raw_features, dst_patch_mask)
        dst_patches_nodes_neighbor_time_features = self.apply_patch_mask(dst_patches_nodes_neighbor_time_features, dst_patch_mask)
        dst_patches_nodes_gcn_features = self.apply_patch_mask(dst_patches_nodes_gcn_features, dst_gcn_patch_mask)

        batch_size = src_patches_nodes_neighbor_node_raw_features.shape[0]
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1
        )
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1
        )
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1
        )
        sequence_patch_mask = torch.cat([src_patch_mask, dst_patch_mask], dim=1)
        sequence_patches_data = torch.stack([
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features
        ], dim=2).reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.sequence_num_channels * self.channel_embedding_dim
        )

        key_padding_mask = sequence_patch_mask == 0
        for transformer in self.transformers:
            sequence_patches_data = transformer(sequence_patches_data, key_padding_mask=key_padding_mask)

        src_sequence_embeddings = self.sequence_output_layer(
            self.masked_mean(sequence_patches_data[:, :src_num_patches, :], src_patch_mask)
        )
        dst_sequence_embeddings = self.sequence_output_layer(
            self.masked_mean(sequence_patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :], dst_patch_mask)
        )

        src_str_embeddings = self.compute_structure_embeddings(
            gcn_patch_features=src_patches_nodes_gcn_features,
            gcn_patch_mask=src_gcn_patch_mask
        )
        dst_str_embeddings = self.compute_structure_embeddings(
            gcn_patch_features=dst_patches_nodes_gcn_features,
            gcn_patch_mask=dst_gcn_patch_mask
        )
        src_str_q, dst_str_q = self.compute_struct_quality(
            pair_gcn_pack=pair_gcn_pack,
            batch_size=batch_size
        )

        src_short_term_embeddings = self.compute_short_term_embeddings(
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features
        )
        dst_short_term_embeddings = self.compute_short_term_embeddings(
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features
        )
        src_seq_embeddings, src_short_aux = self.fuse_long_short(
            src_sequence_embeddings,
            src_short_term_embeddings,
            return_aux=True
        )
        dst_seq_embeddings, dst_short_aux = self.fuse_long_short(
            dst_sequence_embeddings,
            dst_short_term_embeddings,
            return_aux=True
        )

        src_stat, dst_stat, pair_memory, stat_raw_aux = self.compute_stat_embeddings(
            stat_inputs=stat_inputs,
            batch_size=batch_size
        )
        src_seq_embeddings, src_str_embeddings, src_stat, src_str_q = self.apply_channel_masks(
            seq_emb=src_seq_embeddings,
            str_emb=src_str_embeddings,
            stat_emb=src_stat,
            str_q=src_str_q
        )
        dst_seq_embeddings, dst_str_embeddings, dst_stat, dst_str_q = self.apply_channel_masks(
            seq_emb=dst_seq_embeddings,
            str_emb=dst_str_embeddings,
            stat_emb=dst_stat,
            str_q=dst_str_q
        )
        if self.node_fusion_mode == "baseline":
            src_node_embeddings, src_aux = self.node_baseline_fuse(
                seq_emb=src_seq_embeddings,
                str_emb=src_str_embeddings,
                stat_emb=src_stat
            )
            dst_node_embeddings, dst_aux = self.node_baseline_fuse(
                seq_emb=dst_seq_embeddings,
                str_emb=dst_str_embeddings,
                stat_emb=dst_stat
            )
        else:
            src_node_embeddings, src_aux = self.node_ccf(
                seq_emb=src_seq_embeddings,
                str_emb=src_str_embeddings,
                stat_emb=src_stat,
                str_q=src_str_q
            )
            dst_node_embeddings, dst_aux = self.node_ccf(
                seq_emb=dst_seq_embeddings,
                str_emb=dst_str_embeddings,
                stat_emb=dst_stat,
                str_q=dst_str_q
            )
        src_aux["str_q"] = src_str_q
        dst_aux["str_q"] = dst_str_q
        src_aux["valid_anchor_ratio"] = src_str_q[:, 3:4]
        dst_aux["valid_anchor_ratio"] = dst_str_q[:, 3:4]
        src_aux.update(src_short_aux)
        dst_aux.update(dst_short_aux)

        outputs = {
            "src_node": src_node_embeddings,
            "dst_node": dst_node_embeddings,
            "src_aux": src_aux,
            "dst_aux": dst_aux,
            "stat_raw_aux": stat_raw_aux,
        }
        if compute_pair:
            if self.disable_stat or self.pair_memory_mode == "zero":
                pair_memory = torch.zeros_like(pair_memory)
            if self.pair_fusion_mode == "ccf":
                logit, pair_aux = self.pair_ccf(
                    src_node=src_node_embeddings,
                    dst_node=dst_node_embeddings,
                    pair_memory=pair_memory
                )
            else:
                logit, pair_aux = self.pair_plain_fusion(
                    src_node=src_node_embeddings,
                    dst_node=dst_node_embeddings,
                    pair_memory=pair_memory
                )
            pair_aux["pair_memory"] = pair_memory
            pair_aux["memory_direct_gate"] = stat_raw_aux["memory_direct_gate"]
            pair_aux["memory_struct_gate"] = stat_raw_aux["memory_struct_gate"]
            pair_aux["memory_time_gate"] = stat_raw_aux["memory_time_gate"]
            pair_aux["pair_raw"] = stat_raw_aux["pair_raw"]
            pair_aux["pair_time_raw"] = stat_raw_aux["pair_time_raw"]
            outputs["logit"] = logit
            outputs["pair_aux"] = pair_aux
        return outputs

    def _compute_outputs_from_batch(self, batch: dict, compute_pair: bool = True):
        src_padded_nodes_neighbor_ids = batch["src_pad_ids"].cpu().numpy()
        src_padded_nodes_edge_ids = batch["src_pad_eids"].cpu().numpy()
        src_padded_nodes_neighbor_times = batch["src_pad_ts"].cpu().numpy()
        dst_padded_nodes_neighbor_ids = batch["dst_pad_ids"].cpu().numpy()
        dst_padded_nodes_edge_ids = batch["dst_pad_eids"].cpu().numpy()
        dst_padded_nodes_neighbor_times = batch["dst_pad_ts"].cpu().numpy()
        node_interact_times = batch["ts"].cpu().numpy()
        return self._compute_temporal_embeddings_from_padded_inputs(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            src_padded_nodes_edge_ids=src_padded_nodes_edge_ids,
            src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            dst_padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
            dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
            node_interact_times=node_interact_times,
            pair_gcn_pack=batch["gcn"],
            stat_inputs=batch.get("stat"),
            compute_pair=compute_pair
        )

    def forward(self, batch: dict):
        outputs = self._compute_outputs_from_batch(batch=batch, compute_pair=True)
        return {
            "logit": outputs["logit"],
            "src_aux": outputs["src_aux"],
            "dst_aux": outputs["dst_aux"],
            "pair_aux": outputs["pair_aux"],
        }

    def compute_from_batch(self, batch: dict, return_aux: bool = False):
        outputs = self._compute_outputs_from_batch(batch=batch, compute_pair=False)
        if return_aux:
            return outputs["src_node"], outputs["dst_node"], {
                "src_aux": outputs["src_aux"],
                "dst_aux": outputs["dst_aux"],
            }
        return outputs["src_node"], outputs["dst_node"]

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        from utils.collate_gcn import build_batch_from_triplets

        batch = build_batch_from_triplets(
            src=src_node_ids.astype(np.int64),
            dst=dst_node_ids.astype(np.int64),
            ts=node_interact_times.astype(np.float64),
            neighbor_sampler=self.neighbor_sampler,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
            recent_k=self.recent_k,
            max_neighbors=self.max_neighbors_gcn,
            common_neighbors_topk=self.common_neighbors_topk,
            pair_graph_score_alpha=self.pair_graph_score_alpha,
            pair_graph_score_beta=self.pair_graph_score_beta,
            pair_graph_score_gamma=self.pair_graph_score_gamma,
            pair_graph_common_lambda=self.pair_graph_common_lambda,
            anchor_selection_mode=self.anchor_selection_mode,
            edge_construction_mode=self.edge_construction_mode,
            common_neighbor_mode=self.common_neighbor_mode,
            stat_source_mode=self.stat_source_mode,
            stat_decay_mode=self.stat_decay_mode,
            stat_feature_variant=self.stat_feature_variant
        )
        return self.compute_from_batch(batch=batch)

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            max_seq_length = max(max_seq_length, len(nodes_neighbor_ids_list[idx]))
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)
        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            if len(nodes_neighbor_ids_list[idx]) > 0:
                seq_len = len(nodes_neighbor_ids_list[idx])
                padded_nodes_neighbor_ids[idx, 1: seq_len + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: seq_len + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: seq_len + 1] = nodes_neighbor_times_list[idx]
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: 'TimeEncoder'):
        padded_nodes_neighbor_ids_tensor = torch.from_numpy(padded_nodes_neighbor_ids).long()
        padded_nodes_edge_ids_tensor = torch.from_numpy(padded_nodes_edge_ids).long().to(self.device)
        padded_nodes_neighbor_node_raw_features = self.get_node_input_features(padded_nodes_neighbor_ids_tensor)
        padded_nodes_edge_raw_features = self.edge_raw_features[padded_nodes_edge_ids_tensor]
        time_deltas = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=time_deltas)
        zero_mask = torch.from_numpy(padded_nodes_neighbor_ids == 0).to(self.device)
        padded_nodes_neighbor_time_features[zero_mask] = 0.0
        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_gcn_features: torch.Tensor,
                    patch_size: int = 1):
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size
        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_gcn_features_list = []
        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = start_idx + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx:end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx:end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx:end_idx, :])
            patches_nodes_gcn_features_list.append(padded_nodes_gcn_features[:, start_idx:end_idx, :])
        batch_size = padded_nodes_neighbor_node_raw_features.shape[0]
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.node_feat_dim
        )
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.edge_feat_dim
        )
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim
        )
        patches_nodes_gcn_features = torch.stack(patches_nodes_gcn_features_list, dim=1).reshape(
            batch_size, num_patches, patch_size * self.gcn_feat_dim
        )
        return (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_gcn_features
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self.neighbor_sampler, "sample_neighbor_strategy") and \
                self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(attention_dim, 4 * attention_dim),
            nn.Linear(4 * attention_dim, attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = inputs.transpose(0, 1)
        x = self.norm_layers[0](x)
        attn_out = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )[0]
        attn_out = attn_out.transpose(0, 1)
        out = inputs + self.dropout(attn_out)
        ff = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](out)))))
        out = out + self.dropout(ff)
        return out
