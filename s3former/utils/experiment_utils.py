import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.S3Former import S3Former
from models.modules import MergeLayer
from utils.collate_gcn import make_lp_collate_fn
from utils.utils import get_parameter_sizes


def attach_train_context_to_args(args, train_data):
    args.train_seen_node_ids = set(train_data.src_node_ids.tolist()) | set(train_data.dst_node_ids.tolist())
    args.train_cut_time = float(train_data.node_interact_times[-1]) if len(train_data.node_interact_times) > 0 else None
    return args


def get_collate_kwargs_from_args(args):
    return {
        "patch_size": args.patch_size,
        "max_input_sequence_length": args.max_input_sequence_length,
        "recent_k": args.recent_k,
        "max_neighbors": args.max_neighbors_gcn,
        "common_neighbors_topk": args.common_neighbors_topk,
        "pair_graph_score_alpha": args.pair_graph_score_alpha,
        "pair_graph_score_beta": args.pair_graph_score_beta,
        "pair_graph_score_gamma": args.pair_graph_score_gamma,
        "pair_graph_common_lambda": args.pair_graph_common_lambda,
        "anchor_selection_mode": getattr(args, "anchor_selection_mode", "recent_k"),
        "edge_construction_mode": getattr(args, "edge_construction_mode", "sparse_induced"),
        "common_neighbor_mode": getattr(args, "common_neighbor_mode", "topk"),
        "stat_source_mode": getattr(args, "stat_source_mode", "online"),
        "stat_decay_mode": getattr(args, "stat_decay_mode", "fixed"),
        "stat_feature_variant": getattr(args, "stat_feature_variant", "full"),
        "train_seen_node_ids": getattr(args, "train_seen_node_ids", None),
        "train_cut_time": getattr(args, "train_cut_time", None),
    }


def get_model_kwargs_from_args(args):
    return {
        "time_feat_dim": args.time_feat_dim,
        "channel_embedding_dim": args.channel_embedding_dim,
        "patch_size": args.patch_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "max_input_sequence_length": args.max_input_sequence_length,
        "recent_k": args.recent_k,
        "max_neighbors_gcn": args.max_neighbors_gcn,
        "common_neighbors_topk": args.common_neighbors_topk,
        "pair_graph_score_alpha": args.pair_graph_score_alpha,
        "pair_graph_score_beta": args.pair_graph_score_beta,
        "pair_graph_score_gamma": args.pair_graph_score_gamma,
        "pair_graph_common_lambda": args.pair_graph_common_lambda,
        "short_window_size": args.short_window_size,
        "use_short_term": args.use_short_term,
        "short_hidden_dim": args.short_hidden_dim,
        "short_num_layers": args.short_num_layers,
        "short_conv_kernel_size": args.short_conv_kernel_size,
        "short_dropout": args.short_dropout,
        "short_pooling": args.short_pooling,
        "short_fusion_mode": args.short_fusion_mode,
        "pair_ccf_variant": args.pair_ccf_variant,
        "device": args.device,
    }


def create_s3former_collate(data_obj, neighbor_sampler, args):
    return make_lp_collate_fn(
        data_obj=data_obj,
        neighbor_sampler=neighbor_sampler,
        **get_collate_kwargs_from_args(args)
    )


def create_dynamic_backbone(model_name, node_raw_features, edge_raw_features, neighbor_sampler, train_data, args):
    if model_name == 'TGAT':
        return TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
    if model_name in ['JODIE', 'DyRep', 'TGN']:
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
        return MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    if model_name == 'CAWN':
        return CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
    if model_name == 'TCL':
        return TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
    if model_name == 'GraphMixer':
        return GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    if model_name == 'S3Former':
        return S3Former(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                         **get_model_kwargs_from_args(args))
    raise ValueError(f"Wrong value for model_name {model_name}!")


def create_model(model_name, node_raw_features, edge_raw_features, neighbor_sampler, train_data, args):
    dynamic_backbone = create_dynamic_backbone(
        model_name=model_name,
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        train_data=train_data,
        args=args
    )
    if model_name == 'S3Former':
        link_predictor = nn.Identity()
    else:
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
    return nn.Sequential(dynamic_backbone, link_predictor)


def summarize_model_size(model):
    parameter_size = get_parameter_sizes(model) * 4
    return {
        "bytes": parameter_size,
        "kb": parameter_size / 1024,
        "mb": parameter_size / 1024 / 1024,
    }
