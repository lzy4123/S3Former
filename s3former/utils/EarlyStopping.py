import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)

    @staticmethod
    def _get_s3former_backbone(model: nn.Module):
        if hasattr(model, 'use_short_term'):
            return model
        if isinstance(model, nn.Sequential) and len(model) > 0 and hasattr(model[0], 'use_short_term'):
            return model[0]
        return None

    @staticmethod
    def _is_short_term_key(key: str):
        short_term_key_fragments = (
            'short_input_projection',
            'short_conv_layers',
            'short_layer_norms',
            'short_output_layer',
            'short_attention_pooling',
            'short_fusion_gate_layer',
        )
        return any(fragment in key for fragment in short_term_key_fragments)

    def _load_s3former_compatible_checkpoint(self, model: nn.Module, state_dict: dict):
        current_state_dict = model.state_dict()
        compatible_state_dict = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key in current_state_dict and current_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                skipped_keys.append(key)

        incompatible_keys = model.load_state_dict(compatible_state_dict, strict=False)
        self.logger.info(
            "load S3Former checkpoint with compatible-key fallback: "
            f"loaded={len(compatible_state_dict)}, "
            f"missing={len(incompatible_keys.missing_keys)}, "
            f"skipped={len(skipped_keys)}"
        )
        if len(skipped_keys) > 0:
            self.logger.info(f"S3Former skipped checkpoint keys: {skipped_keys[:20]}")
        if len(incompatible_keys.missing_keys) > 0:
            self.logger.info(f"S3Former missing model keys: {incompatible_keys.missing_keys[:20]}")
        if len(incompatible_keys.unexpected_keys) > 0:
            self.logger.info(f"S3Former unexpected model keys: {incompatible_keys.unexpected_keys[:20]}")

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        state_dict = torch.load(self.save_model_path, map_location=map_location)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as strict_error:
            if self.model_name == 'S3Former':
                self._load_s3former_compatible_checkpoint(model=model, state_dict=state_dict)
                if self.model_name in ['JODIE', 'DyRep', 'TGN']:
                    model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
                return

            backbone = self._get_s3former_backbone(model)
            current_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            missing_keys = sorted(current_keys - loaded_keys)
            unexpected_keys = sorted(loaded_keys - current_keys)

            can_try_short_term_fallback = (
                self.model_name == 'S3Former' and
                backbone is not None and
                getattr(backbone, 'use_short_term', False) and
                len(missing_keys) > 0 and
                all(self._is_short_term_key(key) for key in missing_keys) and
                len(unexpected_keys) == 0
            )
            if not can_try_short_term_fallback:
                raise strict_error

            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            if len(incompatible_keys.unexpected_keys) != 0 or \
                    len(incompatible_keys.missing_keys) == 0 or \
                    not all(self._is_short_term_key(key) for key in incompatible_keys.missing_keys):
                raise strict_error

            self.logger.info("load model with S3Former short-term compatibility fallback")
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
