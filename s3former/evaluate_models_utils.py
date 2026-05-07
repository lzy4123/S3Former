import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.collate_gcn import build_batch_from_triplets

def get_link_logits(link_predictor: nn.Module, src_node_embeddings: torch.Tensor, dst_node_embeddings: torch.Tensor):
    """
    Unified forward interface for link prediction heads that output logits.
    """
    return link_predictor(input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1)


def get_link_probabilities(link_predictor: nn.Module, src_node_embeddings: torch.Tensor, dst_node_embeddings: torch.Tensor):
    """
    Unified forward interface for link prediction heads.
    """
    return get_link_logits(
        link_predictor=link_predictor,
        src_node_embeddings=src_node_embeddings,
        dst_node_embeddings=dst_node_embeddings
    ).sigmoid()


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    """
    # Ensures fixed seed for evaluation negatives
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'S3Former']:
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        for batch_idx, batch_in in enumerate(evaluate_idx_data_loader_tqdm):

            # =========================
            # ✅ S3Former + collate batch dict 路径
            # =========================
            if model_name == 'S3Former':
                if not isinstance(batch_in, dict):
                    raise RuntimeError("S3Former must use collate batch dict and compute_from_batch() path during evaluation.")
                batch = batch_in

                # 正样本 embedding
                positive_outputs = model[0](batch)
                positive_logits = positive_outputs["logit"].squeeze(dim=-1)

                # 从 padded 序列第 0 位取中心节点 id & 当前时间
                pos_src = batch["src_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                pos_dst = batch["dst_pad_ids"][:, 0].cpu().numpy().astype(np.int64)
                pos_ts = batch["ts"].cpu().numpy().astype(np.float64)

                # 负采样：random vs 非 random
                if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                    batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                        size=len(pos_src),
                        batch_src_node_ids=pos_src,
                        batch_dst_node_ids=pos_dst,
                        current_batch_start_time=pos_ts[0],
                        current_batch_end_time=pos_ts[-1]
                    )
                else:
                    _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(pos_src))
                    batch_neg_src_node_ids = pos_src

                batch_neg_src_node_ids = batch_neg_src_node_ids.astype(np.int64)
                batch_neg_dst_node_ids = batch_neg_dst_node_ids.astype(np.int64)

                # 构造负样本 batch（注意：评估用的 neighbor_sampler 是 full_neighbor_sampler）
                neg_batch = build_batch_from_triplets(
                    src=batch_neg_src_node_ids,
                    dst=batch_neg_dst_node_ids,
                    ts=pos_ts,
                    neighbor_sampler=neighbor_sampler,
                    patch_size=model[0].patch_size,
                    max_input_sequence_length=model[0].max_input_sequence_length,
                    recent_k=getattr(model[0], "recent_k", 16),
                    max_neighbors=getattr(model[0], "max_neighbors_gcn", 20),
                    common_neighbors_topk=getattr(model[0], "common_neighbors_topk", 8),
                    pair_graph_score_alpha=getattr(model[0], "pair_graph_score_alpha", 1.0),
                    pair_graph_score_beta=getattr(model[0], "pair_graph_score_beta", 1e-6),
                    pair_graph_score_gamma=getattr(model[0], "pair_graph_score_gamma", 1.0),
                    pair_graph_common_lambda=getattr(model[0], "pair_graph_common_lambda", 0.5),
                    anchor_selection_mode=getattr(model[0], "anchor_selection_mode", "recent_k"),
                    edge_construction_mode=getattr(model[0], "edge_construction_mode", "sparse_induced"),
                    common_neighbor_mode=getattr(model[0], "common_neighbor_mode", "topk"),
                    stat_source_mode=getattr(model[0], "stat_source_mode", "online"),
                    stat_decay_mode=getattr(model[0], "stat_decay_mode", "fixed"),
                    stat_feature_variant=getattr(model[0], "stat_feature_variant", "full")
                )

                negative_outputs = model[0](neg_batch)
                negative_logits = negative_outputs["logit"].squeeze(dim=-1)

                logits = torch.cat([positive_logits, negative_logits], dim=0)
                labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)
                loss = loss_func(input=logits, target=labels)
                predicts = logits.sigmoid()

                evaluate_losses.append(loss.item())
                evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')
                continue

            # =========================
            # 原版 indices 路径（其他模型/或 S3Former 未启用 collate）
            # =========================
            evaluate_data_indices = batch_in.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids),
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1]
                )
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors
                    )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors
                    )
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=None,
                        edges_are_positive=False,
                        num_neighbors=num_neighbors
                    )
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        edges_are_positive=True,
                        num_neighbors=num_neighbors
                    )
            elif model_name in ['GraphMixer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors,
                        time_gap=time_gap
                    )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors,
                        time_gap=time_gap
                    )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            positive_logits = get_link_logits(link_predictor=model[1], src_node_embeddings=batch_src_node_embeddings,
                                              dst_node_embeddings=batch_dst_node_embeddings)
            negative_logits = get_link_logits(link_predictor=model[1], src_node_embeddings=batch_neg_src_node_embeddings,
                                              dst_node_embeddings=batch_neg_dst_node_embeddings)

            logits = torch.cat([positive_logits, negative_logits], dim=0)
            labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)
            predicts = logits.sigmoid()

            loss = loss_func(input=logits, target=labels)
            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics



def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'S3Former']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            if model_name == 'S3Former':
                if not isinstance(evaluate_data_indices, dict):
                    raise RuntimeError("S3Former must use collate batch dict and compute_from_batch() path for node classification evaluation.")
                batch = evaluate_data_indices
                batch_src_node_embeddings, _ = model[0].compute_from_batch(batch)

                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
                if "labels" not in batch:
                    raise RuntimeError("S3Former node classification batch must contain labels.")
                labels = batch["labels"].float().to(predicts.device)

                loss = loss_func(input=predicts, target=labels)
                evaluate_total_loss += loss.item()
                evaluate_y_trues.append(labels)
                evaluate_y_predicts.append(predicts)
                evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')
                continue

            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
