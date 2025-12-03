import functools
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import nni

# isort: on
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from compressed_tensors.transform.factory.base import TransformBase
from packaging import version
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_utils import (
    EvalPrediction,
)

from llmcompressor.utils.pytorch.module import (
    patch_module_to_cuda,
    patch_tensor_to_cuda,
    tensor_to_cuda,
)

from .train_utils import SGDG


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        teacher_model = kwargs.pop("teacher_model", None)
        ignored_modules = kwargs.pop("ignored_modules", [])
        self.weight_tied_name_map = kwargs.pop("weight_tied_name_map", {})
        super().__init__(*args, **kwargs)
        if (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        ):
            model: nn.Module = self.model
            # ignored_modules = list()
            torch.distributed.barrier()
            with patch_module_to_cuda(torch.nn.Module):
                for m in ignored_modules:
                    m.cuda()
                for m in model.modules():
                    if isinstance(m, (TransformBase)):
                        ignored_modules.append(m)
                        m.cuda()

            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            self.accelerator.state.fsdp_plugin.use_orig_params = True

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if int(os.environ["RANK"]) == 0:
            nni.report_intermediate_result(loss.item())
        return loss

    @torch.compile(fullgraph=False, disable=True)
    def compute_loss(self, model, inputs, **kwargs):
        args = self.args
        loss_type = args.special.get("loss_type", "origin")
        if loss_type == "origin":
            return super().compute_loss(model, inputs, **kwargs)

        if loss_type == "rkl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss
        if loss_type == "kl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss

        if "r_kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            if loss_type == "k_top":
                k = 1000
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return loss

        if "kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            if loss_type == "kl_top":
                k = 1000
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits * (
                labels != -100
            ).unsqueeze(-1)
            outputs = model(**inputs)
            logits = outputs.logits * (labels != -100).unsqueeze(-1)
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if getattr(args, "post_attn", False):
                ref = F.softmax(ori_logits, dim=-1).gather(-1, indices).flatten(0, -2)
                can = F.log_softmax(logits, dim=-1).gather(-1, indices).flatten(0, -2)
                loss = (
                    F.kl_div(can, ref, reduction="batchmean")
                    * labels.numel()
                    / (labels != -100).sum()
                )
            else:
                top_logits = logits.gather(-1, indices)
                loss = (
                    F.kl_div(
                        F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                        F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    * labels.numel()
                    / (labels != -100).sum()
                )
            return loss

        if loss_type == "mse":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.mse_loss(logits, ori_logits)
            return loss
        if loss_type == "kd":
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            T, alpha = self.temperature, self.loss_alpha
            ori_loss = outputs["loss"]
            logits = logits.view(-1, logits.size(-1))
            ori_logits = ori_logits.view(-1, ori_logits.size(-1))
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
            return loss
        if loss_type == "DFT":
            shift_labels = inputs.pop("labels", None)[:, 1:].contiguous().flatten()
            outputs = model(**inputs)
            shift_logits = outputs.logits[:, :-1].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            loss = (
                loss
                * F.softmax(shift_logits, dim=-1)
                .gather(1, shift_labels.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            ).mean()
            return loss

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        inputs = dict(inputs)
        inputs.pop("labels", None)

        outputs = model.teacher(**inputs, output_hidden_states=True)
        model.teacher._is_root = False
        return outputs

    @classmethod
    def register_tied_parameters(cls, model, weight_tied_name_map):
        for names, tied_names in weight_tied_name_map.items():
            if names != tied_names:
                (module_name, param_name) = names
                module = model.get_submodule(module_name)
                module.register_parameter(
                    param_name,
                    model.get_parameter(f"{tied_names[0]}.{tied_names[1]}"),
                )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        import geoopt
        from geoopt.manifolds import EuclideanStiefel, Stiefel

        for m in self.accelerator.state.fsdp_plugin.ignored_modules:
            for name, param in m.named_parameters(recurse=False):
                if param.requires_grad and len(param.size()) > 1:
                    m.register_parameter(
                        name, geoopt.ManifoldParameter(param.data, manifold=Stiefel())
                    )
        self.register_tied_parameters(self.model, self.weight_tied_name_map)

        args = self.args
        params_rotate = []
        params_smooth = []
        for param in self.model.parameters():
            param: torch.nn.Parameter
            if param.requires_grad:
                if len(param.size()) == 1:
                    params_smooth.append(param)
                else:
                    params_rotate.append(param)
        dict_rotate = {
            "params": params_rotate,
            "lr": args.special.get("rotate_lr", 0.1),
            "momentum": args.special.get("rotate_momentum", 0.0),
            "stiefel": True,
            "grassmann": True,
            "omega": 0.1,
        }
        dict_smooth = {
            "params": params_smooth,
            "lr": args.special.get("smooth_lr", 0.0),
            "momentum": args.special.get("smooth_momentum", 0.0),
            "stiefel": False,
            "nesterov": False,
        }
        if args.special.get("opt_type", "SGDG") == "SGDG":
            optimizer = SGDG([dict_rotate, dict_smooth], weight_decay=0)
        elif args.special.get("opt_type", "SGDG") == "RSGD":
            import geoopt

            optimizer = geoopt.optim.RiemannianSGD(
                [dict_rotate, dict_smooth],
                weight_decay=0,
                lr=args.special.get("rotate_lr", 0.1),
                stabilize=10,
            )
        elif args.special.get("opt_type", "SGDG") == "RAdam":
            import geoopt

            optimizer = geoopt.optim.RiemannianAdam(
                [dict_rotate, dict_smooth],
                weight_decay=0,
                lr=args.special.get("rotate_lr", 0.1),
                stabilize=10,
            )
        self.optimizer = optimizer

        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )

    def get_trained_params(self):
        """
        Returns a copy of the model on CPU.
        """
        if self.is_fsdp_enabled:
            state_dict = pt_fsdp_state_dict(self.model)
            return state_dict
        else:
            return self.model.state_dict()
