import numpy as np
import torch


ROLE_FEAT_DIM = 10
EDGE_ATTR_DIM = 5
OLD_OLD = 0
OLD_NEW = 1
NEW_NEW = 2


def pad_sequences_np(node_ids, node_times, neigh_ids_list, edge_ids_list, neigh_times_list,
                     patch_size: int, max_input_sequence_length: int):
    max_seq_length = 0
    for i in range(len(neigh_ids_list)):
        if len(neigh_ids_list[i]) > max_input_sequence_length - 1:
            neigh_ids_list[i] = neigh_ids_list[i][-(max_input_sequence_length - 1):]
            edge_ids_list[i] = edge_ids_list[i][-(max_input_sequence_length - 1):]
            neigh_times_list[i] = neigh_times_list[i][-(max_input_sequence_length - 1):]
        max_seq_length = max(max_seq_length, len(neigh_ids_list[i]))

    max_seq_length += 1
    if max_seq_length % patch_size != 0:
        max_seq_length += (patch_size - max_seq_length % patch_size)

    batch_size = len(node_ids)
    padded_ids = np.zeros((batch_size, max_seq_length), dtype=np.int64)
    padded_eids = np.zeros((batch_size, max_seq_length), dtype=np.int64)
    padded_ts = np.zeros((batch_size, max_seq_length), dtype=np.float32)

    for i in range(batch_size):
        padded_ids[i, 0] = int(node_ids[i])
        padded_ts[i, 0] = float(node_times[i])
        if len(neigh_ids_list[i]) > 0:
            seq_len = len(neigh_ids_list[i])
            padded_ids[i, 1: 1 + seq_len] = np.asarray(neigh_ids_list[i], dtype=np.int64)
            padded_eids[i, 1: 1 + seq_len] = np.asarray(edge_ids_list[i], dtype=np.int64)
            padded_ts[i, 1: 1 + seq_len] = np.asarray(neigh_times_list[i], dtype=np.float32)

    return padded_ids, padded_eids, padded_ts


def _stable_random_state(*values):
    seed = 0
    for value in values:
        seed = ((seed * 1315423911) ^ int(abs(hash(value)))) & 0xFFFFFFFF
    return np.random.RandomState(seed)


def _summarize_neighbor_history(neighbor_ids: np.ndarray, neighbor_times: np.ndarray, current_time: float):
    summary = {}
    for neighbor_id, neighbor_time in zip(neighbor_ids.tolist(), neighbor_times.tolist()):
        neighbor_id = int(neighbor_id)
        if neighbor_id == 0:
            continue
        neighbor_time = float(neighbor_time)
        if neighbor_id not in summary:
            summary[neighbor_id] = {"count": 0, "last_time": neighbor_time}
        summary[neighbor_id]["count"] += 1
        summary[neighbor_id]["last_time"] = neighbor_time

    for stats in summary.values():
        stats["delta_t"] = max(float(current_time) - float(stats["last_time"]), 0.0)
    return summary


def _compute_node_score(stats: dict, score_alpha: float, score_beta: float, score_gamma: float):
    if stats is None:
        return 0.0
    return float(
        score_alpha * np.exp(-score_beta * stats["delta_t"]) +
        score_gamma * np.log1p(stats["count"])
    )


def _select_anchor_neighbors(summary: dict, topk: int, score_alpha: float, score_beta: float, score_gamma: float,
                             selection_mode: str, node_id: int, current_time: float):
    if len(summary) == 0:
        return [], {}

    scored_neighbors = []
    score_map = {}
    for neighbor_id, stats in summary.items():
        score = _compute_node_score(stats=stats, score_alpha=score_alpha, score_beta=score_beta, score_gamma=score_gamma)
        score_map[int(neighbor_id)] = score
        scored_neighbors.append((int(neighbor_id), score, float(stats["last_time"]), int(stats["count"])))

    scored_neighbors.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    if selection_mode == "all_history" or topk <= 0:
        return [neighbor_id for neighbor_id, _, _, _ in scored_neighbors], score_map
    if selection_mode == "random_k":
        random_state = _stable_random_state(node_id, int(current_time * 1000.0), topk, len(scored_neighbors))
        sampled_idx = random_state.choice(len(scored_neighbors), size=min(topk, len(scored_neighbors)), replace=False)
        sampled = [scored_neighbors[int(i)] for i in sampled_idx.tolist()]
        sampled.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        return [neighbor_id for neighbor_id, _, _, _ in sampled], score_map
    return [neighbor_id for neighbor_id, _, _, _ in scored_neighbors[:topk]], score_map


def _build_role_feature(node_id: int, src_node_id: int, dst_node_id: int,
                        src_summary: dict, dst_summary: dict,
                        src_recent_set: set, dst_recent_set: set, common_neighbor_set: set,
                        score_beta: float):
    src_stats = src_summary.get(int(node_id))
    dst_stats = dst_summary.get(int(node_id))

    src_recency = float(np.exp(-score_beta * src_stats["delta_t"])) if src_stats is not None else 0.0
    dst_recency = float(np.exp(-score_beta * dst_stats["delta_t"])) if dst_stats is not None else 0.0
    src_count = float(np.log1p(src_stats["count"])) if src_stats is not None else 0.0
    dst_count = float(np.log1p(dst_stats["count"])) if dst_stats is not None else 0.0

    return np.asarray([
        1.0 if int(node_id) == int(src_node_id) else 0.0,
        1.0 if int(node_id) == int(dst_node_id) else 0.0,
        1.0 if int(node_id) in src_recent_set else 0.0,
        1.0 if int(node_id) in dst_recent_set else 0.0,
        1.0 if int(node_id) in common_neighbor_set else 0.0,
        0.0,
        src_recency,
        dst_recency,
        src_count,
        dst_count,
    ], dtype=np.float32)


def _append_candidate(candidate_nodes: list, candidate_seen: set, node_id: int):
    node_id = int(node_id)
    if node_id == 0 or node_id in candidate_seen:
        return
    candidate_nodes.append(node_id)
    candidate_seen.add(node_id)


def _node_old_new_type(src_node_id: int, dst_node_id: int, train_seen_node_ids):
    if train_seen_node_ids is None:
        return OLD_OLD
    src_old = int(src_node_id) in train_seen_node_ids
    dst_old = int(dst_node_id) in train_seen_node_ids
    if src_old and dst_old:
        return OLD_OLD
    if src_old or dst_old:
        return OLD_NEW
    return NEW_NEW


def _build_candidate_nodes(src_node_id: int, dst_node_id: int, current_time: float, neighbor_sampler,
                           recent_k: int, common_neighbors_topk: int,
                           score_alpha: float, score_beta: float, score_gamma: float, common_lambda: float,
                           anchor_selection_mode: str, common_neighbor_mode: str):
    src_neighbor_ids, _, src_neighbor_times, _ = neighbor_sampler.find_neighbors_before(int(src_node_id), float(current_time))
    dst_neighbor_ids, _, dst_neighbor_times, _ = neighbor_sampler.find_neighbors_before(int(dst_node_id), float(current_time))

    src_summary = _summarize_neighbor_history(src_neighbor_ids, src_neighbor_times, current_time=float(current_time))
    dst_summary = _summarize_neighbor_history(dst_neighbor_ids, dst_neighbor_times, current_time=float(current_time))

    src_recent_neighbors, src_score_map = _select_anchor_neighbors(
        summary=src_summary,
        topk=int(recent_k),
        score_alpha=score_alpha,
        score_beta=score_beta,
        score_gamma=score_gamma,
        selection_mode=anchor_selection_mode,
        node_id=src_node_id,
        current_time=current_time
    )
    dst_recent_neighbors, dst_score_map = _select_anchor_neighbors(
        summary=dst_summary,
        topk=int(recent_k),
        score_alpha=score_alpha,
        score_beta=score_beta,
        score_gamma=score_gamma,
        selection_mode=anchor_selection_mode,
        node_id=dst_node_id,
        current_time=current_time
    )

    common_neighbor_candidates = []
    all_common_neighbor_ids = set(src_summary.keys()) & set(dst_summary.keys())
    for neighbor_id in all_common_neighbor_ids:
        score_u = src_score_map.get(int(neighbor_id), _compute_node_score(src_summary[int(neighbor_id)], score_alpha, score_beta, score_gamma))
        score_v = dst_score_map.get(int(neighbor_id), _compute_node_score(dst_summary[int(neighbor_id)], score_alpha, score_beta, score_gamma))
        common_score = score_u + score_v + common_lambda * min(score_u, score_v)
        latest_time = max(float(src_summary[int(neighbor_id)]["last_time"]), float(dst_summary[int(neighbor_id)]["last_time"]))
        common_neighbor_candidates.append((int(neighbor_id), float(common_score), latest_time))

    common_neighbor_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
    if common_neighbor_mode == "off":
        common_neighbors = []
    else:
        common_neighbors = [neighbor_id for neighbor_id, _, _ in common_neighbor_candidates[:int(common_neighbors_topk)]]

    candidate_nodes, candidate_seen = [], set()
    _append_candidate(candidate_nodes, candidate_seen, src_node_id)
    _append_candidate(candidate_nodes, candidate_seen, dst_node_id)
    for neighbor_id in src_recent_neighbors:
        _append_candidate(candidate_nodes, candidate_seen, neighbor_id)
    for neighbor_id in dst_recent_neighbors:
        _append_candidate(candidate_nodes, candidate_seen, neighbor_id)
    for neighbor_id in common_neighbors:
        _append_candidate(candidate_nodes, candidate_seen, neighbor_id)

    return {
        "candidate_nodes": candidate_nodes,
        "src_summary": src_summary,
        "dst_summary": dst_summary,
        "src_recent_neighbors": src_recent_neighbors,
        "dst_recent_neighbors": dst_recent_neighbors,
        "common_neighbors": common_neighbors,
        "all_common_neighbor_count": float(len(all_common_neighbor_ids)),
        "recent_common_neighbor_count": float(len(set(src_recent_neighbors) & set(dst_recent_neighbors))),
    }


def _resolve_decay_betas(score_beta: float, stat_decay_mode: str):
    if stat_decay_mode == "none":
        return 0.0, 0.0
    if stat_decay_mode == "pair_only":
        return 0.0, float(score_beta)
    return float(score_beta), float(score_beta)


def _history_before(neighbor_sampler, node_id: int, current_time: float, stat_source_mode: str):
    query_time = float("inf") if stat_source_mode == "static_global" else float(current_time)
    neighbor_ids, _, neighbor_times, _ = neighbor_sampler.find_neighbors_before(int(node_id), query_time)
    return neighbor_ids, neighbor_times


def _node_raw_stat_from_history(neighbor_ids: np.ndarray, neighbor_times: np.ndarray, current_time: float, score_beta: float):
    valid_mask = neighbor_ids != 0
    neighbor_ids = neighbor_ids[valid_mask]
    neighbor_times = neighbor_times[valid_mask]
    history_count = float(len(neighbor_ids))
    unique_neighbors = float(len(set(neighbor_ids.tolist()))) if len(neighbor_ids) > 0 else 0.0
    if len(neighbor_times) == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    deltas = np.maximum(float(current_time) - neighbor_times.astype(np.float64), 0.0)
    latest_recency = float(np.exp(-score_beta * float(deltas.min()))) if score_beta > 0 else float(history_count > 0)
    decayed_history = float(np.exp(-score_beta * deltas).sum()) if score_beta > 0 else history_count
    return np.asarray([
        float(np.log1p(history_count)),
        float(np.log1p(unique_neighbors)),
        latest_recency,
        float(np.log1p(decayed_history)),
    ], dtype=np.float32)


def _apply_stat_feature_variant(pair_raw: np.ndarray, pair_time_raw: np.ndarray, stat_feature_variant: str):
    if stat_feature_variant == "full":
        return pair_raw, pair_time_raw
    pair_raw = pair_raw.copy()
    pair_time_raw = pair_time_raw.copy()
    if stat_feature_variant == "no_stability":
        pair_raw[:, 3:8] = 0.0
    elif stat_feature_variant == "no_overlap":
        pair_raw[:, 8:16] = 0.0
    elif stat_feature_variant == "no_pair_time":
        pair_time_raw[:, :] = 0.0
    elif stat_feature_variant == "direct_only":
        pair_raw[:, 8:16] = 0.0
        pair_time_raw[:, 1:] = 0.0
    return pair_raw, pair_time_raw


def _pair_raw_and_time_stats(src_pad_ids_np: np.ndarray, src_pad_ts_np: np.ndarray,
                             dst_pad_ids_np: np.ndarray, dst_pad_ts_np: np.ndarray,
                             src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                             neighbor_sampler, recent_k: int, score_beta: float,
                             stat_source_mode: str, stat_decay_mode: str, stat_feature_variant: str):
    batch_size = len(ts)
    src_raw = np.zeros((batch_size, 4), dtype=np.float32)
    dst_raw = np.zeros((batch_size, 4), dtype=np.float32)
    pair_raw = np.zeros((batch_size, 16), dtype=np.float32)
    pair_time_raw = np.zeros((batch_size, 8), dtype=np.float32)

    node_beta, pair_beta = _resolve_decay_betas(score_beta=score_beta, stat_decay_mode=stat_decay_mode)
    sampler_times = [times for times in getattr(neighbor_sampler, "nodes_neighbor_times", []) if len(times) > 0]
    if len(sampler_times) > 0:
        min_time = float(min(times[0] for times in sampler_times))
        max_time = float(max(times[-1] for times in sampler_times))
    else:
        min_time = float(np.min(ts)) if batch_size > 0 else 0.0
        max_time = float(np.max(ts)) if batch_size > 0 else 1.0
    time_span = max(max_time - min_time, 1.0)
    if time_span <= 64.0:
        phase_period = 12.0
    elif time_span >= 2.0 * 365.0 * 24.0 * 3600.0:
        phase_period = 365.0 * 24.0 * 3600.0
    else:
        phase_period = max(time_span / 12.0, 1.0)

    for idx in range(batch_size):
        current_time = float(ts[idx])
        src_ids_hist, src_times_hist = _history_before(neighbor_sampler, int(src[idx]), current_time, stat_source_mode)
        dst_ids_hist, dst_times_hist = _history_before(neighbor_sampler, int(dst[idx]), current_time, stat_source_mode)
        src_raw[idx] = _node_raw_stat_from_history(src_ids_hist, src_times_hist, current_time=current_time, score_beta=node_beta)
        dst_raw[idx] = _node_raw_stat_from_history(dst_ids_hist, dst_times_hist, current_time=current_time, score_beta=node_beta)

        src_ids = src_ids_hist[src_ids_hist != 0]
        src_times = src_times_hist[:len(src_ids)] if len(src_times_hist) != len(src_ids_hist) else src_times_hist[src_ids_hist != 0]
        dst_ids = dst_ids_hist[dst_ids_hist != 0]
        dst_times = dst_times_hist[:len(dst_ids)] if len(dst_times_hist) != len(dst_ids_hist) else dst_times_hist[dst_ids_hist != 0]

        src_recent_ids = src_ids[-recent_k:] if recent_k > 0 else src_ids[:0]
        dst_recent_ids = dst_ids[-recent_k:] if recent_k > 0 else dst_ids[:0]
        src_recent_set = set(src_recent_ids.tolist())
        dst_recent_set = set(dst_recent_ids.tolist())
        recent_overlap = float(len(src_recent_set & dst_recent_set))
        recent_union = float(len(src_recent_set | dst_recent_set))
        recent_jaccard = recent_overlap / recent_union if recent_union > 0 else 0.0

        src_total_set = set(src_ids.tolist())
        dst_total_set = set(dst_ids.tolist())
        common_neighbors = src_total_set & dst_total_set
        total_overlap = float(len(common_neighbors))

        decayed_cooccurrence = 0.0
        latest_shared_time = None
        for neighbor_id in common_neighbors:
            src_neighbor_times = src_times[src_ids == neighbor_id]
            dst_neighbor_times = dst_times[dst_ids == neighbor_id]
            shared_count = float(min(len(src_neighbor_times), len(dst_neighbor_times)))
            neighbor_latest_time = max(float(src_neighbor_times.max()), float(dst_neighbor_times.max()))
            delta_t = max(current_time - neighbor_latest_time, 0.0)
            weight = np.exp(-pair_beta * delta_t) if pair_beta > 0 else 1.0
            decayed_cooccurrence += shared_count * weight
            latest_shared_time = neighbor_latest_time if latest_shared_time is None else max(latest_shared_time, neighbor_latest_time)

        query_time = float("inf") if stat_source_mode == "static_global" else current_time
        direct_stats = neighbor_sampler.get_pair_statistics_before(
            node_i=int(src[idx]),
            node_j=int(dst[idx]),
            interact_time=query_time,
            beta=pair_beta
        )
        direct_pair_times = neighbor_sampler.get_pair_history_before(
            node_i=int(src[idx]),
            node_j=int(dst[idx]),
            interact_time=query_time,
            directed=False
        )
        long_direct_count = float(direct_stats["count"])
        long_direct_decay = float(direct_stats["decay_sum"])
        recent_direct_times = direct_pair_times[-recent_k:] if recent_k > 0 else direct_pair_times[:0]
        recent_direct_count = float(len(recent_direct_times))
        if recent_direct_count > 0:
            recent_direct_deltas = np.maximum(current_time - recent_direct_times.astype(np.float64), 0.0)
            recent_direct_decay = float(np.exp(-pair_beta * recent_direct_deltas).sum()) if pair_beta > 0 else recent_direct_count
        else:
            recent_direct_decay = 0.0

        direct_recency = 0.0
        direct_gap_norm = 1.0
        if direct_stats["count"] > 0:
            direct_delta = max(current_time - min(float(direct_stats["last_time"]), current_time), 0.0)
            direct_recency = float(np.exp(-pair_beta * direct_delta)) if pair_beta > 0 else 1.0
            direct_gap_norm = float(np.log1p(direct_delta) / np.log1p(time_span + 1.0))

        pair_raw[idx] = np.asarray([
            float(np.log1p(long_direct_count)),
            float(np.log1p(long_direct_decay)),
            float(direct_stats["bidirectional_flag"]),
            long_direct_decay / max(long_direct_count, 1.0),
            float(np.log1p(recent_direct_count)),
            float(np.log1p(recent_direct_decay)),
            recent_direct_count / max(long_direct_count, 1.0),
            recent_direct_decay / max(long_direct_decay, 1e-6),
            float(np.log1p(recent_overlap)),
            float(recent_jaccard),
            float(np.log1p(total_overlap)),
            recent_overlap / max(total_overlap, 1.0),
            float(np.log1p(decayed_cooccurrence)),
            decayed_cooccurrence / max(total_overlap, 1.0),
            float(np.log1p(total_overlap / max(long_direct_count, 1.0))),
            float(np.log1p(recent_overlap / max(recent_direct_count, 1.0))),
        ], dtype=np.float32)

        shared_recency = 0.0
        if latest_shared_time is not None:
            shared_delta = max(current_time - latest_shared_time, 0.0)
            shared_recency = float(np.exp(-pair_beta * shared_delta)) if pair_beta > 0 else 1.0

        normalized_time = float(np.clip((current_time - min_time) / time_span, 0.0, 1.0))
        phase_angle = 2.0 * np.pi * ((current_time - min_time) / phase_period)

        pair_time_raw[idx] = np.asarray([
            direct_recency,
            shared_recency,
            float(src_raw[idx, 2]),
            float(dst_raw[idx, 2]),
            normalized_time,
            float(np.sin(phase_angle)),
            float(np.cos(phase_angle)),
            direct_gap_norm,
        ], dtype=np.float32)

    pair_raw, pair_time_raw = _apply_stat_feature_variant(pair_raw, pair_time_raw, stat_feature_variant=stat_feature_variant)
    return src_raw, dst_raw, pair_raw, pair_time_raw


def _iter_edge_sources(edge_construction_mode: str, candidate_nodes: list, src_node_id: int, dst_node_id: int):
    if edge_construction_mode == "one_hop_only":
        return [int(src_node_id), int(dst_node_id)]
    return [int(node_id) for node_id in candidate_nodes]


def _collect_neighbor_ids_for_edges(neighbor_sampler, node_id: int, current_time: float,
                                    edge_construction_mode: str, max_neighbors: int):
    neighbor_ids, _, _, _ = neighbor_sampler.find_neighbors_before(node_id=int(node_id), interact_time=float(current_time))
    if edge_construction_mode in {"sparse_induced", "one_hop_only", "one_hop_induced"} and max_neighbors > 0 and len(neighbor_ids) > max_neighbors:
        neighbor_ids = neighbor_ids[-max_neighbors:]
    return np.unique(neighbor_ids)


def build_pair_graph_pack(src_pad_ids_np: np.ndarray, dst_pad_ids_np: np.ndarray,
                          src_pad_ts_np: np.ndarray, dst_pad_ts_np: np.ndarray,
                          src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                          neighbor_sampler, recent_k: int, max_neighbors: int,
                          common_neighbors_topk: int, score_alpha: float,
                          score_beta: float, score_gamma: float, common_lambda: float,
                          anchor_selection_mode: str = "recent_k",
                          edge_construction_mode: str = "sparse_induced",
                          common_neighbor_mode: str = "topk",
                          train_seen_node_ids=None,
                          add_self_loop: bool = True):
    batch_size, src_seq_len = src_pad_ids_np.shape
    _, dst_seq_len = dst_pad_ids_np.shape

    uniq_nodes_all_parts = []
    node_role_feature_parts = []
    edge_src_all, edge_dst_all, edge_attr_all = [], [], []
    src_token_map = -np.ones((batch_size, src_seq_len), dtype=np.int64)
    dst_token_map = -np.ones((batch_size, dst_seq_len), dtype=np.int64)
    node_counts = np.zeros((batch_size,), dtype=np.float32)
    edge_counts = np.zeros((batch_size,), dtype=np.float32)
    densities = np.zeros((batch_size,), dtype=np.float32)
    src_valid_anchor_ratios = np.zeros((batch_size,), dtype=np.float32)
    dst_valid_anchor_ratios = np.zeros((batch_size,), dtype=np.float32)
    common_neighbor_counts = np.zeros((batch_size,), dtype=np.float32)
    recent_common_neighbor_counts = np.zeros((batch_size,), dtype=np.float32)
    triangle_closure_scores = np.zeros((batch_size,), dtype=np.float32)
    two_hop_closure_flags = np.zeros((batch_size,), dtype=np.float32)
    old_old_new_types = np.zeros((batch_size,), dtype=np.int64)

    node_offset = 0
    for batch_idx in range(batch_size):
        src_node_id = int(src[batch_idx])
        dst_node_id = int(dst[batch_idx])
        current_time = float(ts[batch_idx])

        candidate_info = _build_candidate_nodes(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            current_time=current_time,
            neighbor_sampler=neighbor_sampler,
            recent_k=recent_k,
            common_neighbors_topk=common_neighbors_topk,
            score_alpha=score_alpha,
            score_beta=score_beta,
            score_gamma=score_gamma,
            common_lambda=common_lambda,
            anchor_selection_mode=anchor_selection_mode,
            common_neighbor_mode=common_neighbor_mode
        )
        candidate_nodes = candidate_info["candidate_nodes"]
        src_summary = candidate_info["src_summary"]
        dst_summary = candidate_info["dst_summary"]
        src_recent_set = set(candidate_info["src_recent_neighbors"])
        dst_recent_set = set(candidate_info["dst_recent_neighbors"])
        common_neighbor_set = set(candidate_info["common_neighbors"])
        common_neighbor_counts[batch_idx] = float(candidate_info["all_common_neighbor_count"])
        recent_common_neighbor_counts[batch_idx] = float(candidate_info["recent_common_neighbor_count"])
        two_hop_closure_flags[batch_idx] = 1.0 if common_neighbor_counts[batch_idx] > 0 else 0.0
        old_old_new_types[batch_idx] = _node_old_new_type(src_node_id, dst_node_id, train_seen_node_ids=train_seen_node_ids)

        if len(candidate_nodes) == 0:
            continue

        local_id_to_idx = {int(node_id): local_idx for local_idx, node_id in enumerate(candidate_nodes)}
        uniq_nodes_all_parts.append(np.asarray(candidate_nodes, dtype=np.int64))
        node_role_feature_parts.append(np.stack([
            _build_role_feature(
                node_id=node_id,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                src_summary=src_summary,
                dst_summary=dst_summary,
                src_recent_set=src_recent_set,
                dst_recent_set=dst_recent_set,
                common_neighbor_set=common_neighbor_set,
                score_beta=score_beta
            )
            for node_id in candidate_nodes
        ], axis=0))

        added_undirected_pairs = set()
        sample_edge_count = 0
        for node_id in _iter_edge_sources(edge_construction_mode=edge_construction_mode,
                                          candidate_nodes=candidate_nodes,
                                          src_node_id=src_node_id,
                                          dst_node_id=dst_node_id):
            local_idx = local_id_to_idx[node_id]
            if add_self_loop:
                edge_src_all.append(node_offset + local_idx)
                edge_dst_all.append(node_offset + local_idx)
                edge_attr_all.append(np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

            neighbor_ids = _collect_neighbor_ids_for_edges(
                neighbor_sampler=neighbor_sampler,
                node_id=node_id,
                current_time=current_time,
                edge_construction_mode=edge_construction_mode,
                max_neighbors=max_neighbors
            )
            for neighbor_id in neighbor_ids:
                neighbor_id = int(neighbor_id)
                if neighbor_id == node_id or neighbor_id not in local_id_to_idx:
                    continue
                if edge_construction_mode == "one_hop_only" and node_id not in {src_node_id, dst_node_id}:
                    continue
                undirected_pair = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if undirected_pair in added_undirected_pairs:
                    continue
                if edge_construction_mode == "one_hop_only" and src_node_id not in undirected_pair and dst_node_id not in undirected_pair:
                    continue
                added_undirected_pairs.add(undirected_pair)

                pair_stats = neighbor_sampler.get_pair_statistics_before(
                    node_i=node_id,
                    node_j=neighbor_id,
                    interact_time=current_time,
                    beta=score_beta
                )
                if pair_stats["count"] <= 0:
                    continue

                local_neighbor_idx = local_id_to_idx[neighbor_id]
                edge_feature = np.asarray([
                    float(np.log1p(pair_stats["count"])),
                    float(np.exp(-score_beta * max(current_time - pair_stats["last_time"], 0.0))),
                    float(pair_stats["decay_sum"]),
                    float(pair_stats["bidirectional_flag"]),
                    0.0,
                ], dtype=np.float32)
                edge_src_all.append(node_offset + local_idx)
                edge_dst_all.append(node_offset + local_neighbor_idx)
                edge_attr_all.append(edge_feature)
                edge_src_all.append(node_offset + local_neighbor_idx)
                edge_dst_all.append(node_offset + local_idx)
                edge_attr_all.append(edge_feature.copy())
                sample_edge_count += 1

        for seq_idx, node_id in enumerate(src_pad_ids_np[batch_idx]):
            node_id = int(node_id)
            if node_id != 0 and node_id in local_id_to_idx:
                src_token_map[batch_idx, seq_idx] = node_offset + local_id_to_idx[node_id]
        for seq_idx, node_id in enumerate(dst_pad_ids_np[batch_idx]):
            node_id = int(node_id)
            if node_id != 0 and node_id in local_id_to_idx:
                dst_token_map[batch_idx, seq_idx] = node_offset + local_id_to_idx[node_id]

        num_nodes = float(len(candidate_nodes))
        node_counts[batch_idx] = num_nodes
        edge_counts[batch_idx] = float(sample_edge_count)
        densities[batch_idx] = float(sample_edge_count) / (num_nodes * (num_nodes - 1.0) / 2.0 + 1e-6)
        anchor_denom = max(float(recent_k), 1.0)
        src_valid_anchor_ratios[batch_idx] = min(float(len(src_recent_set)) / anchor_denom, 1.0) if anchor_selection_mode != "all_history" else 1.0
        dst_valid_anchor_ratios[batch_idx] = min(float(len(dst_recent_set)) / anchor_denom, 1.0) if anchor_selection_mode != "all_history" else 1.0
        triangle_closure_scores[batch_idx] = float(np.log1p(common_neighbor_counts[batch_idx] + edge_counts[batch_idx] * densities[batch_idx]))
        node_offset += len(candidate_nodes)

    if len(uniq_nodes_all_parts) == 0:
        return {
            "uniq_nodes": np.zeros((0,), dtype=np.int64),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32),
            "node_role_features": np.zeros((0, ROLE_FEAT_DIM), dtype=np.float32),
            "src_token_map": src_token_map,
            "dst_token_map": dst_token_map,
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "densities": densities,
            "src_valid_anchor_ratios": src_valid_anchor_ratios,
            "dst_valid_anchor_ratios": dst_valid_anchor_ratios,
            "common_neighbor_counts": common_neighbor_counts,
            "recent_common_neighbor_counts": recent_common_neighbor_counts,
            "triangle_closure_scores": triangle_closure_scores,
            "two_hop_closure_flags": two_hop_closure_flags,
            "old_old_new_types": old_old_new_types,
        }

    uniq_nodes_all = np.concatenate(uniq_nodes_all_parts, axis=0)
    node_role_features = np.concatenate(node_role_feature_parts, axis=0).astype(np.float32)
    if len(edge_src_all) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32)
    else:
        edge_index = np.stack([np.asarray(edge_src_all, dtype=np.int64), np.asarray(edge_dst_all, dtype=np.int64)], axis=0)
        edge_attr = np.stack(edge_attr_all, axis=0).astype(np.float32)

    return {
        "uniq_nodes": uniq_nodes_all,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_role_features": node_role_features,
        "src_token_map": src_token_map,
        "dst_token_map": dst_token_map,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "densities": densities,
        "src_valid_anchor_ratios": src_valid_anchor_ratios,
        "dst_valid_anchor_ratios": dst_valid_anchor_ratios,
        "common_neighbor_counts": common_neighbor_counts,
        "recent_common_neighbor_counts": recent_common_neighbor_counts,
        "triangle_closure_scores": triangle_closure_scores,
        "two_hop_closure_flags": two_hop_closure_flags,
        "old_old_new_types": old_old_new_types,
    }


def pair_graph_pack_to_torch(pair_graph_pack: dict):
    return {
        "pair": {
            "uniq_nodes": torch.from_numpy(pair_graph_pack["uniq_nodes"]).long(),
            "edge_index": torch.from_numpy(pair_graph_pack["edge_index"]).long(),
            "edge_attr": torch.from_numpy(pair_graph_pack["edge_attr"]).float(),
            "node_role_features": torch.from_numpy(pair_graph_pack["node_role_features"]).float(),
            "src_token_map": torch.from_numpy(pair_graph_pack["src_token_map"]).long(),
            "dst_token_map": torch.from_numpy(pair_graph_pack["dst_token_map"]).long(),
            "node_counts": torch.from_numpy(pair_graph_pack["node_counts"]).float(),
            "edge_counts": torch.from_numpy(pair_graph_pack["edge_counts"]).float(),
            "densities": torch.from_numpy(pair_graph_pack["densities"]).float(),
            "src_valid_anchor_ratios": torch.from_numpy(pair_graph_pack["src_valid_anchor_ratios"]).float(),
            "dst_valid_anchor_ratios": torch.from_numpy(pair_graph_pack["dst_valid_anchor_ratios"]).float(),
            "common_neighbor_counts": torch.from_numpy(pair_graph_pack["common_neighbor_counts"]).float(),
            "recent_common_neighbor_counts": torch.from_numpy(pair_graph_pack["recent_common_neighbor_counts"]).float(),
            "triangle_closure_scores": torch.from_numpy(pair_graph_pack["triangle_closure_scores"]).float(),
            "two_hop_closure_flags": torch.from_numpy(pair_graph_pack["two_hop_closure_flags"]).float(),
            "old_old_new_types": torch.from_numpy(pair_graph_pack["old_old_new_types"]).long(),
        }
    }


def build_batch_from_triplets(src, dst, ts, neighbor_sampler,
                              patch_size, max_input_sequence_length,
                              recent_k, max_neighbors,
                              common_neighbors_topk,
                              pair_graph_score_alpha,
                              pair_graph_score_beta,
                              pair_graph_score_gamma,
                              pair_graph_common_lambda,
                              anchor_selection_mode: str = "recent_k",
                              edge_construction_mode: str = "sparse_induced",
                              common_neighbor_mode: str = "topk",
                              stat_source_mode: str = "online",
                              stat_decay_mode: str = "fixed",
                              stat_feature_variant: str = "full",
                              train_seen_node_ids=None,
                              train_cut_time=None):
    src_nids_list, src_eids_list, src_nts_list = neighbor_sampler.get_all_first_hop_neighbors(
        node_ids=src, node_interact_times=ts
    )
    dst_nids_list, dst_eids_list, dst_nts_list = neighbor_sampler.get_all_first_hop_neighbors(
        node_ids=dst, node_interact_times=ts
    )

    src_pad_ids, src_pad_eids, src_pad_ts = pad_sequences_np(
        src, ts, src_nids_list, src_eids_list, src_nts_list,
        patch_size=patch_size, max_input_sequence_length=max_input_sequence_length
    )
    dst_pad_ids, dst_pad_eids, dst_pad_ts = pad_sequences_np(
        dst, ts, dst_nids_list, dst_eids_list, dst_nts_list,
        patch_size=patch_size, max_input_sequence_length=max_input_sequence_length
    )

    pair_graph_pack = build_pair_graph_pack(
        src_pad_ids_np=src_pad_ids,
        dst_pad_ids_np=dst_pad_ids,
        src_pad_ts_np=src_pad_ts,
        dst_pad_ts_np=dst_pad_ts,
        src=src,
        dst=dst,
        ts=ts,
        neighbor_sampler=neighbor_sampler,
        recent_k=recent_k,
        max_neighbors=max_neighbors,
        common_neighbors_topk=common_neighbors_topk,
        score_alpha=pair_graph_score_alpha,
        score_beta=pair_graph_score_beta,
        score_gamma=pair_graph_score_gamma,
        common_lambda=pair_graph_common_lambda,
        anchor_selection_mode=anchor_selection_mode,
        edge_construction_mode=edge_construction_mode,
        common_neighbor_mode=common_neighbor_mode,
        train_seen_node_ids=train_seen_node_ids,
        add_self_loop=True
    )
    src_raw_stat, dst_raw_stat, pair_raw_stat, pair_time_raw = _pair_raw_and_time_stats(
        src_pad_ids_np=src_pad_ids,
        src_pad_ts_np=src_pad_ts,
        dst_pad_ids_np=dst_pad_ids,
        dst_pad_ts_np=dst_pad_ts,
        src=src,
        dst=dst,
        ts=ts,
        neighbor_sampler=neighbor_sampler,
        recent_k=recent_k,
        score_beta=pair_graph_score_beta,
        stat_source_mode=stat_source_mode,
        stat_decay_mode=stat_decay_mode,
        stat_feature_variant=stat_feature_variant
    )

    src_pad_ids_tensor = torch.from_numpy(src_pad_ids).long()
    src_pad_eids_tensor = torch.from_numpy(src_pad_eids).long()
    src_pad_ts_tensor = torch.from_numpy(src_pad_ts).float()
    dst_pad_ids_tensor = torch.from_numpy(dst_pad_ids).long()
    dst_pad_eids_tensor = torch.from_numpy(dst_pad_eids).long()
    dst_pad_ts_tensor = torch.from_numpy(dst_pad_ts).float()
    ts_tensor = torch.from_numpy(ts.astype(np.float32)).float()
    batch = {
        "src_pad_ids": src_pad_ids_tensor,
        "src_pad_eids": src_pad_eids_tensor,
        "src_pad_ts": src_pad_ts_tensor,
        "dst_pad_ids": dst_pad_ids_tensor,
        "dst_pad_eids": dst_pad_eids_tensor,
        "dst_pad_ts": dst_pad_ts_tensor,
        "ts": ts_tensor,
        "src_ids": torch.from_numpy(src.astype(np.int64)).long(),
        "dst_ids": torch.from_numpy(dst.astype(np.int64)).long(),
        "times": ts_tensor,
        "seq": {
            "src_pad_ids": src_pad_ids_tensor,
            "src_pad_eids": src_pad_eids_tensor,
            "src_pad_ts": src_pad_ts_tensor,
            "dst_pad_ids": dst_pad_ids_tensor,
            "dst_pad_eids": dst_pad_eids_tensor,
            "dst_pad_ts": dst_pad_ts_tensor,
        },
        "gcn": pair_graph_pack_to_torch(pair_graph_pack),
        "stat": {
            "src_raw": torch.from_numpy(src_raw_stat).float(),
            "dst_raw": torch.from_numpy(dst_raw_stat).float(),
            "pair_raw": torch.from_numpy(pair_raw_stat).float(),
            "pair_time_raw": torch.from_numpy(pair_time_raw).float(),
        },
    }
    if train_cut_time is not None:
        batch["train_cut_time"] = torch.tensor(float(train_cut_time), dtype=torch.float32)
    return batch


def make_lp_collate_fn(data_obj, neighbor_sampler,
                       patch_size: int, max_input_sequence_length: int,
                       recent_k: int, max_neighbors: int,
                       common_neighbors_topk: int,
                       pair_graph_score_alpha: float,
                       pair_graph_score_beta: float,
                       pair_graph_score_gamma: float,
                       pair_graph_common_lambda: float,
                       anchor_selection_mode: str = "recent_k",
                       edge_construction_mode: str = "sparse_induced",
                       common_neighbor_mode: str = "topk",
                       stat_source_mode: str = "online",
                       stat_decay_mode: str = "fixed",
                       stat_feature_variant: str = "full",
                       train_seen_node_ids=None,
                       train_cut_time=None):
    def _collate(indices):
        idx = np.asarray(indices, dtype=np.int64)

        src = data_obj.src_node_ids[idx].astype(np.int64)
        dst = data_obj.dst_node_ids[idx].astype(np.int64)
        ts = data_obj.node_interact_times[idx].astype(np.float64)
        edge_ids = data_obj.edge_ids[idx].astype(np.int64) if hasattr(data_obj, "edge_ids") else None
        labels = data_obj.labels[idx].astype(np.float32) if hasattr(data_obj, "labels") else None

        batch = build_batch_from_triplets(
            src=src,
            dst=dst,
            ts=ts,
            neighbor_sampler=neighbor_sampler,
            patch_size=patch_size,
            max_input_sequence_length=max_input_sequence_length,
            recent_k=recent_k,
            max_neighbors=max_neighbors,
            common_neighbors_topk=common_neighbors_topk,
            pair_graph_score_alpha=pair_graph_score_alpha,
            pair_graph_score_beta=pair_graph_score_beta,
            pair_graph_score_gamma=pair_graph_score_gamma,
            pair_graph_common_lambda=pair_graph_common_lambda,
            anchor_selection_mode=anchor_selection_mode,
            edge_construction_mode=edge_construction_mode,
            common_neighbor_mode=common_neighbor_mode,
            stat_source_mode=stat_source_mode,
            stat_decay_mode=stat_decay_mode,
            stat_feature_variant=stat_feature_variant,
            train_seen_node_ids=train_seen_node_ids,
            train_cut_time=train_cut_time
        )

        if edge_ids is not None:
            batch["edge_ids"] = torch.from_numpy(edge_ids).long()
        if labels is not None:
            batch["labels"] = torch.from_numpy(labels).float()
        return batch

    return _collate
