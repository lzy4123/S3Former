import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from utils.collate_gcn import build_batch_from_triplets
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, get_link_logits
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.experiment_utils import attach_train_context_to_args, create_s3former_collate, create_model


def _cuda_device_for_args(args):
    if not torch.cuda.is_available() or not str(args.device).startswith('cuda'):
        return None
    return torch.device(args.device)


def _profile_sync(args):
    device = _cuda_device_for_args(args)
    if device is not None:
        torch.cuda.synchronize(device)


def _profile_reset_peak_memory(args):
    device = _cuda_device_for_args(args)
    if device is not None:
        torch.cuda.reset_peak_memory_stats(device)


def _profile_peak_memory_gb(args):
    device = _cuda_device_for_args(args)
    if device is None:
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def _log_efficiency(logger, message):
    line = f'[Efficiency] {message}'
    logger.info(line)
    print(line, flush=True)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    attach_train_context_to_args(args, train_data=train_data)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    # get data loaders
    if args.model_name == "S3Former":
        train_collate = create_s3former_collate(data_obj=train_data, neighbor_sampler=train_neighbor_sampler, args=args)
        val_collate = create_s3former_collate(data_obj=val_data, neighbor_sampler=full_neighbor_sampler, args=args)
        new_node_val_collate = create_s3former_collate(data_obj=new_node_val_data, neighbor_sampler=full_neighbor_sampler, args=args)
        test_collate = create_s3former_collate(data_obj=test_data, neighbor_sampler=full_neighbor_sampler, args=args)
        new_node_test_collate = create_s3former_collate(data_obj=new_node_test_data, neighbor_sampler=full_neighbor_sampler, args=args)

        # 先 num_workers=0 跑通，再改 2/4
        train_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(train_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=train_collate,
            num_workers=0,
            pin_memory=False
        )
        val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(val_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=0,
            pin_memory=False
        )
        new_node_val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_val_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=new_node_val_collate,
            num_workers=0,
            pin_memory=False
        )
        test_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(test_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_collate,
            num_workers=0,
            pin_memory=False
        )
        new_node_test_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_test_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=new_node_test_collate,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                    batch_size=args.batch_size, shuffle=False)
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                                  batch_size=args.batch_size, shuffle=False)
        new_node_val_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                                   batch_size=args.batch_size, shuffle=False)
        new_node_test_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        model = create_model(model_name=args.model_name,
                             node_raw_features=node_raw_features,
                             edge_raw_features=edge_raw_features,
                             neighbor_sampler=train_neighbor_sampler,
                             train_data=train_data,
                             args=args)
        logger.info(f'model -> {model}')
        num_parameters = get_parameter_sizes(model)
        logger.info(f'model name: {args.model_name}, #parameters: {num_parameters * 4} B, '
                    f'{num_parameters * 4 / 1024} KB, {num_parameters * 4 / 1024 / 1024} MB.')
        if args.profile_efficiency:
            profile_model_name = 'S3Former' if args.model_name == 'S3Former' else args.model_name
            _log_efficiency(
                logger=logger,
                message=f'dataset={args.dataset_name} model={profile_model_name} params={num_parameters / 1e6:.2f}M'
            )

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()
        efficiency_epoch_times, efficiency_epoch_peak_mems = [], []
        efficiency_test_inference_time, efficiency_eval_peak_mem, efficiency_throughput = None, None, None

        for epoch in range(args.num_epochs):

            if args.profile_efficiency:
                _profile_sync(args)
                _profile_reset_peak_memory(args)
                efficiency_epoch_start_time = time.time()

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'S3Former']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                # ===================== ✅ S3Former：batch 是 dict，先处理并 continue =====================
                if args.model_name == "S3Former":
                    batch = train_data_indices
                    if not isinstance(batch, dict):
                        raise RuntimeError("S3Former must use collate batch dict and compute_from_batch() path during training.")
                    # ===== LeakCheck (src side) =====
                    if epoch == 0 and batch_idx == 0:  # 只检查第一个 batch
                        pos_src = batch["src_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                        pos_dst = batch["dst_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                        pos_ts = batch["ts"].cpu().numpy().astype(np.float64)

                        leak_cnt = 0
                        leak_examples = []

                        # 只抽查 200 条（你之前也是 200）
                        check_n = min(200, len(pos_src))
                        for i in range(check_n):
                            s = int(pos_src[i])
                            d = int(pos_dst[i])
                            t0 = float(pos_ts[i])

                            # 关键：用 t0 查“历史”邻居
                            neigh_ids, _, neigh_ts, _ = train_neighbor_sampler.find_neighbors_before(s, t0)

                            # neigh_ids / neigh_ts 建议转 np.array 防止是 list
                            neigh_ids = np.asarray(neigh_ids, dtype=np.int64)
                            neigh_ts = np.asarray(neigh_ts, dtype=np.float64)

                            hit = (neigh_ids == d)
                            if hit.any():
                                times = neigh_ts[hit]
                                # 真泄露判定：历史边时间 >= t0
                                if (times >= t0).any():
                                    leak_cnt += 1
                                    leak_examples.append((i, s, d, t0, times.min(), times.max()))

                        print(f"[LeakCheck] TRUE-leak count: {leak_cnt} / {check_n}")
                        for ex in leak_examples[:10]:
                            i, s, d, t0, tmin, tmax = ex
                            print(f"[LeakCheck] i={i} src={s} dst={d} t0={t0} hit_times(min,max)=({tmin},{tmax})")
                    # ===== LeakCheck end =====
                    # 正样本
                    positive_outputs = model[0](batch)
                    positive_logits = positive_outputs["logit"].squeeze(dim=-1)

                    # 负样本：只换 dst
                    pos_src = batch["src_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                    pos_dst = batch["dst_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                    pos_ts = batch["ts"].cpu().numpy().astype(np.float64)
                    neg_batch_ts = np.nextafter(pos_ts, -np.inf)

                    _, neg_dst = train_neg_edge_sampler.sample(size=len(pos_src))
                    neg_dst = neg_dst.astype(np.int64)

                    neg_batch = build_batch_from_triplets(
                        src=pos_src,
                        dst=neg_dst,
                        ts=neg_batch_ts,
                        neighbor_sampler=train_neighbor_sampler,
                        patch_size=args.patch_size,
                        max_input_sequence_length=args.max_input_sequence_length,
                        recent_k=args.recent_k,
                        max_neighbors=args.max_neighbors_gcn,
                        common_neighbors_topk=args.common_neighbors_topk,
                        pair_graph_score_alpha=args.pair_graph_score_alpha,
                        pair_graph_score_beta=args.pair_graph_score_beta,
                        pair_graph_score_gamma=args.pair_graph_score_gamma,
                        pair_graph_common_lambda=args.pair_graph_common_lambda,
                        anchor_selection_mode=getattr(args, "anchor_selection_mode", "recent_k"),
                        edge_construction_mode=getattr(args, "edge_construction_mode", "sparse_induced"),
                        common_neighbor_mode=getattr(args, "common_neighbor_mode", "topk"),
                        stat_source_mode=getattr(args, "stat_source_mode", "online"),
                        stat_decay_mode=getattr(args, "stat_decay_mode", "fixed"),
                        stat_feature_variant=getattr(args, "stat_feature_variant", "full"),
                        train_seen_node_ids=getattr(args, "train_seen_node_ids", None),
                        train_cut_time=getattr(args, "train_cut_time", None)
                    )
                    negative_outputs = model[0](neg_batch)
                    negative_logits = negative_outputs["logit"].squeeze(dim=-1)

                    logits = torch.cat([positive_logits, negative_logits], dim=0)
                    labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)
                    loss = loss_func(input=logits, target=labels)
                    predicts = logits.sigmoid()

                    train_losses.append(loss.item())
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                    if batch_idx % 100 == 0:
                        print(
                            "[CCF] "
                            f"src alpha={positive_outputs['src_aux']['alpha'].mean().item():.4f}, "
                            f"src gate={positive_outputs['src_aux']['gate'].mean().item():.4f}, "
                            f"dst alpha={positive_outputs['dst_aux']['alpha'].mean().item():.4f}, "
                            f"pair beta={positive_outputs['pair_aux']['beta'].mean().item():.4f}, "
                            f"pair gamma={positive_outputs['pair_aux']['gamma'].mean().item():.4f}/"
                            f"{positive_outputs['pair_aux']['gamma'].std(unbiased=False).item():.4f}, "
                            f"memory gates="
                            f"{positive_outputs['pair_aux']['memory_direct_gate'].mean().item():.4f}/"
                            f"{positive_outputs['pair_aux']['memory_struct_gate'].mean().item():.4f}/"
                            f"{positive_outputs['pair_aux']['memory_time_gate'].mean().item():.4f}, "
                            f"memory bias={positive_outputs['pair_aux']['memory_bias'].mean().item():.4f}, "
                            f"src short eta={positive_outputs['src_aux']['short_eta'].mean().item():.4f}/"
                            f"{positive_outputs['src_aux']['short_eta'].std(unbiased=False).item():.4f}, "
                            f"src short/long="
                            f"{positive_outputs['src_aux']['short_norm'].mean().item():.4f}/"
                            f"{positive_outputs['src_aux']['long_norm'].mean().item():.4f}, "
                            f"dst short eta={positive_outputs['dst_aux']['short_eta'].mean().item():.4f}/"
                            f"{positive_outputs['dst_aux']['short_eta'].std(unbiased=False).item():.4f}"
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_idx_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}'
                    )

                    continue  # ✅ 关键：跳过下面所有旧的 indices 切片逻辑

                # ===================== 其他模型：旧的 indices 路径 =====================
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            #     if batch_idx == 0:
            #         print(type(train_data_indices))
            #         print(train_data_indices.keys())
            #         print(train_data_indices["src_pad_ids"].shape)
            #         print(train_data_indices["gcn"]["src"]["uniq_nodes"].shape)
            #
            #     # train_data_indices = train_data_indices.numpy()
            #     batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
            #         train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
            #         train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
            #
            #     _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            #     batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                # elif args.model_name in ['S3Former']:
                #     # get temporal embedding of source and destination nodes
                #     # two Tensors, with shape (batch_size, node_feat_dim)
                #     batch_src_node_embeddings, batch_dst_node_embeddings = \
                #         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                #                                                           dst_node_ids=batch_dst_node_ids,
                #                                                           node_interact_times=batch_node_interact_times)
                #
                #     # get temporal embedding of negative source and negative destination nodes
                #     # two Tensors, with shape (batch_size, node_feat_dim)
                #     batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                #         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                #                                                           dst_node_ids=batch_neg_dst_node_ids,
                #                                                           node_interact_times=batch_node_interact_times)
                # elif args.model_name in ['S3Former']:
                #     # 此时 train_data_indices 实际上是 collate_fn 返回的 batch dict
                #     batch = train_data_indices
                #
                #     # 正样本：直接走 compute_from_batch
                #     batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_from_batch(batch)
                #
                #     # 从 padded 序列第 0 位拿中心节点 id（CPU numpy）
                #     pos_src = batch["src_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                #     pos_ts = batch["ts"].cpu().numpy().astype(np.float64)
                #
                #     # 负采样：只替换 dst（保持 src 不变）
                #     _, neg_dst = train_neg_edge_sampler.sample(size=len(pos_src))
                #     neg_dst = neg_dst.astype(np.int64)
                #
                #     # 用同一个 sampler（train_neighbor_sampler）构造负样本 batch dict
                #     # recent_k/max_neighbors 要与上面 collate 一致
                #     neg_batch = build_batch_from_triplets(
                #         src=pos_src,
                #         dst=neg_dst,
                #         ts=pos_ts,
                #         neighbor_sampler=train_neighbor_sampler,
                #         patch_size=args.patch_size,
                #         max_input_sequence_length=args.max_input_sequence_length,
                #         recent_k=16,
                #         max_neighbors=20
                #     )
                #
                #     batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_from_batch(
                #         neg_batch)

                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_logits = get_link_logits(link_predictor=model[1], src_node_embeddings=batch_src_node_embeddings,
                                                  dst_node_embeddings=batch_dst_node_embeddings)
                negative_logits = get_link_logits(link_predictor=model[1], src_node_embeddings=batch_neg_src_node_embeddings,
                                                  dst_node_embeddings=batch_neg_dst_node_embeddings)

                logits = torch.cat([positive_logits, negative_logits], dim=0)
                labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)
                loss = loss_func(input=logits, target=labels)
                predicts = logits.sigmoid()

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            if args.profile_efficiency:
                _profile_sync(args)
                epoch_time = time.time() - efficiency_epoch_start_time
                epoch_peak_mem = _profile_peak_memory_gb(args)
                efficiency_epoch_times.append(epoch_time)
                efficiency_epoch_peak_mems.append(epoch_peak_mem)
                _log_efficiency(
                    logger=logger,
                    message=f'epoch={epoch + 1} time={epoch_time:.2f}s peak_mem={epoch_peak_mem:.2f}GB'
                )

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                # reload training memory bank for new validation nodes
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        if args.profile_efficiency:
            _profile_sync(args)
            _profile_reset_peak_memory(args)
            efficiency_test_start_time = time.time()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        if args.profile_efficiency:
            _profile_sync(args)
            efficiency_test_inference_time = time.time() - efficiency_test_start_time
            efficiency_eval_peak_mem = _profile_peak_memory_gb(args)
            num_test_edges = len(test_data.src_node_ids)
            efficiency_throughput = num_test_edges / efficiency_test_inference_time if efficiency_test_inference_time > 0 else 0.0

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reload validation memory bank for new testing nodes
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        if args.profile_efficiency:
            avg_epoch_time = float(np.mean(efficiency_epoch_times[1:] if len(efficiency_epoch_times) > 1 else efficiency_epoch_times))
            peak_train_mem = float(np.max(efficiency_epoch_peak_mems)) if len(efficiency_epoch_peak_mems) > 0 else 0.0
            trans_ap = test_metric_dict.get('average_precision', 0.0)
            _log_efficiency(logger=logger, message=f'avg_epoch_time_exclude_first={avg_epoch_time:.2f}s')
            _log_efficiency(logger=logger, message=f'peak_train_mem={peak_train_mem:.2f}GB')
            _log_efficiency(logger=logger, message=f'test_inference_time={efficiency_test_inference_time:.2f}s')
            _log_efficiency(logger=logger, message=f'eval_peak_mem={efficiency_eval_peak_mem:.2f}GB')
            _log_efficiency(logger=logger, message=f'throughput={efficiency_throughput:.2f} edges/s')
            _log_efficiency(logger=logger, message=f'trans_ap={trans_ap:.4f}')

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        if args.profile_efficiency:
            result_json["efficiency"] = {
                "epoch_times": [f'{epoch_time:.2f}' for epoch_time in efficiency_epoch_times],
                "epoch_peak_mems_gb": [f'{peak_mem:.2f}' for peak_mem in efficiency_epoch_peak_mems],
                "avg_epoch_time_exclude_first": f'{avg_epoch_time:.2f}',
                "peak_train_mem_gb": f'{peak_train_mem:.2f}',
                "test_inference_time": f'{efficiency_test_inference_time:.2f}',
                "eval_peak_mem_gb": f'{efficiency_eval_peak_mem:.2f}',
                "throughput_edges_per_second": f'{efficiency_throughput:.2f}',
                "trans_ap": f'{test_metric_dict.get("average_precision", 0.0):.4f}'
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
            logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
