# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
yyh: add rts for my gate

The file has been adapted from two fairscale files:
 (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
 (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
 Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
 We retain the following license from the original files:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from .mappings import drop_tokens, gather_tokens

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            # TODO: replace with DS process group
            group: torch.distributed.ProcessGroup,
            input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _capacity_plus_on(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1] - 1
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False,
               placeholder_expert: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    if placeholder_expert:
        capacity = _capacity_plus_on(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))
    else:
        capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        capacity = new_capacity

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    if placeholder_expert:
        mask1 = mask1[:, 1:]
        gates = gates[:, 1:]
        
    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        raise Exception("yyh here use_tutel")
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    expert_not_full_ratio = (locations1[-1] < capacity - 1).sum() / locations1.shape[1]

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    if placeholder_expert:
        skip_ratio = (indices1_s == 0).sum() / indices1_s.shape[0]
        
        gate_info = {
        "expert_not_full_ratio": expert_not_full_ratio,
        "skip_ratio": skip_ratio,
                 }
    else:
        gate_info = {
            "expert_not_full_ratio": expert_not_full_ratio
                    }

    return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def old_thresholdGating(logits: Tensor, capacity_factor: float, min_capacity: int, k: int, threshold: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    top1_p, _ = torch.max(gates, dim=1)

    ### devloping ###
    one_expert_token_num = (top1_p > threshold).sum()
    #################

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # threshold
    gates_sorted, gates_indices = torch.sort(gates, dim=-1, descending=True)
    
    cum_sorted_gates = torch.cumsum(gates_sorted, dim=-1)
    chosen_flag = (cum_sorted_gates - gates_sorted) < threshold  # (token num, expert num)
    chosen_flag[:, 0] = True # at least select one expert
    whole_chosen_indices = chosen_flag * (gates_indices + 1) # (token num, expert_num) \in (0, expert_num + 1), 0 means not choose
    chosen_indices = torch.transpose(whole_chosen_indices, 0, 1)  # (expert_num, token_num)

    all_mask = []
    # for i in range(0, chosen_indices.shape[0]):
    for i in range(0, int(2 * capacity_factor)):
        if chosen_indices[i].sum() == 0:
            break

        temp_mask = F.one_hot(chosen_indices[i], num_classes=num_experts + 1) # (token num, expert_num + 1)
        all_mask.append(temp_mask[:, 1:]) # (token num, expert_num)

    dynamic_k = len(all_mask)

    # Compute locations in capacity buffer
    all_locations = [torch.cumsum(all_mask[0], dim=0) - 1]
    location_offset_matrix = torch.sum(all_mask[0], dim=0, keepdim=True)
    for i in range(1, dynamic_k):
        this_locations = torch.cumsum(all_mask[i], dim=0) - 1
        all_locations.append(this_locations + location_offset_matrix)

        location_offset_matrix += torch.sum(all_mask[i], dim=0, keepdim=True)

    # gating decisions (no use)
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
    me = torch.mean(gates, dim=0)
    
    # ce = torch.sum(mask1.float(), dim=0)
    # for i in range(1, dynamic_k):
    #     ce += torch.sum(all_mask[i].float(), dim=0)
    # ce /= ce.sum()
    location_offset_matrix = location_offset_matrix.float().squeeze(0)
    ce = location_offset_matrix / location_offset_matrix.sum()

    # ce = torch.mean(mask1.float(), dim=0)

    l_aux = torch.sum(me * ce) * num_experts

    # Calculate combine_weights and dispatch_mask
    combine_weights = None

    all_valid_chosen_num = 0.0
    for i in range(dynamic_k):
        # Remove locations outside capacity from mask
        all_mask[i] = all_mask[i] * torch.lt(all_locations[i], capacity)
        
        this_valid_chosen_num = (all_mask[i] > 0).sum()
        all_valid_chosen_num += this_valid_chosen_num

        this_mask_float = all_mask[i].float()
        this_gates = gates * this_mask_float # (s, e)

        # Store the capacity location for each token
        this_locations_sc = _one_hot_to_float(torch.sum(all_locations[i] * all_mask[i], dim=1), capacity) # (s, c)
        this_combine_sec = einsum("se,sc->sec", this_gates, this_locations_sc) # (s, e, c)

        if i == 0:
            combine_weights = this_combine_sec
        else:
            combine_weights += this_combine_sec

    
    avg_valid_chosen_num = all_valid_chosen_num / gates.shape[0]

    dispatch_mask = combine_weights.bool()
    
    return l_aux, combine_weights, dispatch_mask, exp_counts, top1_p, avg_valid_chosen_num


def plus_one_thresholdGating(logits: Tensor, capacity_factor: float, min_capacity: int, k: int, threshold: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    top1_p, _ = torch.max(gates, dim=1)

    ### devloping ###
    # one_expert_token_num = (top1_p > threshold).sum()
    #################

    capacity = _capacity_plus_on(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    num_experts = int(gates.shape[1])
    
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions (no use)
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # threshold
    # gates_sorted, gates_indices = torch.sort(gates, dim=-1, descending=True)
    explore_top_k_num = max(min(int(2 * k), num_experts), 1)
    gates_sorted, gates_indices = torch.topk(gates, dim=-1, k=explore_top_k_num, largest=True, sorted=True)
    
    cum_sorted_gates = torch.cumsum(gates_sorted, dim=-1)
    chosen_flag = (cum_sorted_gates - gates_sorted) < threshold  # (token num, explore_top_k_num)
    chosen_flag[:, 0] = True # at least select one expert
    whole_chosen_indices = chosen_flag * (gates_indices + 1) # (token num, explore_top_k_num) \in (0, actual_expert_num), 0 means not choose, 1 means choose placeholder expert (the first expert)
    skip_ratio = (whole_chosen_indices == 1).sum() / whole_chosen_indices.shape[0]

    chosen_indices = torch.transpose(whole_chosen_indices, 0, 1)  # (explore_top_k_num, token_num)

    # get masks and capacity locations
    # stop_index = chosen_indices.sum(dim=-1).ne(0).sum()
    # stop_index = min(int(2 * capacity_factor), stop_index)

    # chosen_indices = chosen_indices[:stop_index] # (chosen_num, token_num)

    tensor_all_mask = F.one_hot(chosen_indices, num_classes=num_experts + 1)[:, :, 2:] # (explore_top_k_num, token_num, expert_num)
    _, token_num, expert_num = tensor_all_mask.shape

    tensor_all_mask = tensor_all_mask.view(-1, expert_num) # (chosen_num * token_num, expert_num)

    # random token selection (ignore position)
    top_idx = _top_idx(tensor_all_mask, capacity)
    new_mask1 = tensor_all_mask * torch.zeros_like(tensor_all_mask).scatter_(0, top_idx, 1)
    tensor_all_mask = new_mask1
    # -------------------------------------------

    tensor_all_locations = torch.cumsum(tensor_all_mask, dim=0).view(-1, token_num, expert_num) - 1 # (chosen_num, token_num, expert_num)
    tensor_all_mask = tensor_all_mask.reshape(-1, token_num, expert_num)    
    
    expert_received_num = tensor_all_locations[-1, -1, :] + 1
    expert_not_full_ratio = (expert_received_num < capacity).sum() / expert_num # maybe near 100% since placeholder expert
    
    tensor_all_locations = tensor_all_locations * tensor_all_mask # (chosen_num, token_num, expert_num)

    # Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
    me = torch.mean(gates, dim=0)
    
    # ce = torch.sum(mask1.float(), dim=0)
    # for i in range(1, dynamic_k):
    #     ce += torch.sum(all_mask[i].float(), dim=0)
    # ce /= ce.sum()

    # ce = expert_received_num / expert_received_num.sum()

    ce = torch.mean(mask1.float(), dim=0)

    l_aux = torch.sum(me * ce) * num_experts
    
    gates = gates[:, 1:]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????


    tensor_all_mask = tensor_all_mask.sum(dim=0) # (token_num, expert_num)
    
    token_chosen_num = tensor_all_mask.sum(dim=-1) # (token_num)
    token_not_full_ratio = (token_chosen_num < capacity_factor).sum() / token_num
    token_no_expert_ratio = (token_chosen_num == 0).sum() / token_num

    tensor_all_locations = tensor_all_locations.sum(dim=0) # (token_num, expert_num)
    # tensor_all_mask = tensor_all_mask * torch.lt(tensor_all_locations, capacity)
    
    tensor_all_mask_float = tensor_all_mask.float()
    tensor_all_gates = gates * tensor_all_mask_float # (s, e)


    all_locations_sc = _one_hot_to_float(tensor_all_locations, capacity) # (s, e, c)
    combine_weights = all_locations_sc * tensor_all_gates.unsqueeze(-1) # (s, e, c)

    dispatch_mask = combine_weights.bool()
    
    all_valid_chosen_num = tensor_all_mask.sum()
    avg_valid_chosen_num = all_valid_chosen_num / token_num

    gate_info = {
        "top1_p": top1_p,
        "chosen_num": avg_valid_chosen_num,
        "skip_ratio": skip_ratio,
        "token_not_full_ratio": token_not_full_ratio,
        "token_no_choose_ratio": token_no_expert_ratio,
        "expert_not_full_ratio": expert_not_full_ratio
                 }
    return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info


def new_thresholdGating(logits: Tensor, capacity_factor: float, min_capacity: int, k: int, threshold: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    top1_p, _ = torch.max(gates, dim=1)

    ### devloping ###
    # one_expert_token_num = (top1_p > threshold).sum()
    #################

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions (no use)
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # threshold
    # gates_sorted, gates_indices = torch.sort(gates, dim=-1, descending=True)
    top_k_num = max(min(int(2 * k), num_experts), 1)
    gates_sorted, gates_indices = torch.topk(gates, dim=-1, k=top_k_num, largest=True, sorted=True)
    
    cum_sorted_gates = torch.cumsum(gates_sorted, dim=-1)
    chosen_flag = (cum_sorted_gates - gates_sorted) < threshold  # (token num, expert num)
    chosen_flag[:, 0] = True # at least select one expert
    whole_chosen_indices = chosen_flag * (gates_indices + 1) # (token num, expert_num) \in (0, expert_num + 1), 0 means not choose
    chosen_indices = torch.transpose(whole_chosen_indices, 0, 1)  # (expert_num, token_num)

    # get masks and capacity locations
    stop_index = chosen_indices.sum(dim=-1).ne(0).sum()
    stop_index = min(int(2 * capacity_factor), stop_index)

    chosen_indices = chosen_indices[:stop_index] # (chosen_num, token_num)
    tensor_all_mask = F.one_hot(chosen_indices, num_classes=num_experts + 1)[:, :, 1:] # (chosen_num, token_num, expert_num)
    _, token_num, expert_num = tensor_all_mask.shape

    tensor_all_mask = tensor_all_mask.view(-1, expert_num) # (chosen_num * token_num, expert_num)

    # random token selection (ignore position)
    top_idx = _top_idx(tensor_all_mask, capacity)
    new_mask1 = tensor_all_mask * torch.zeros_like(tensor_all_mask).scatter_(0, top_idx, 1)
    tensor_all_mask = new_mask1
    # -------------------------------------------

    tensor_all_locations = torch.cumsum(tensor_all_mask, dim=0).view(-1, token_num, expert_num) - 1 # (chosen_num, token_num, expert_num)
    tensor_all_mask = tensor_all_mask.reshape(-1, token_num, expert_num)
    
    expert_received_num = tensor_all_locations[-1, -1, :] + 1
    expert_not_full_ratio = (expert_received_num < capacity).sum() / expert_num
    
    tensor_all_locations = tensor_all_locations * tensor_all_mask # (chosen_num, token_num, expert_num)

    # Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
    me = torch.mean(gates, dim=0)
    
    # ce = torch.sum(mask1.float(), dim=0)
    # for i in range(1, dynamic_k):
    #     ce += torch.sum(all_mask[i].float(), dim=0)
    # ce /= ce.sum()

    # ce = expert_received_num / expert_received_num.sum()

    ce = torch.mean(mask1.float(), dim=0)

    l_aux = torch.sum(me * ce) * num_experts
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????


    tensor_all_mask = tensor_all_mask.sum(dim=0) # (token_num, expert_num)
    
    token_chosen_num = tensor_all_mask.sum(dim=-1) # (token_num)
    token_not_full_ratio = (token_chosen_num < capacity_factor).sum() / token_num

    tensor_all_locations = tensor_all_locations.sum(dim=0) # (token_num, expert_num)
    # tensor_all_mask = tensor_all_mask * torch.lt(tensor_all_locations, capacity)
    
    tensor_all_mask_float = tensor_all_mask.float()
    tensor_all_gates = gates * tensor_all_mask_float # (s, e)

    all_locations_sc = _one_hot_to_float(tensor_all_locations, capacity) # (s, e, c)
    combine_weights = all_locations_sc * tensor_all_gates.unsqueeze(-1) # (s, e, c)

    dispatch_mask = combine_weights.bool()

    
    all_valid_chosen_num = tensor_all_mask.sum()
    avg_valid_chosen_num = all_valid_chosen_num / token_num

    gate_info = {
        "top1_p": top1_p,
        "chosen_num": avg_valid_chosen_num,
        "token_not_full_ratio": token_not_full_ratio,
        "expert_not_full_ratio": expert_not_full_ratio
                 }
    return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 threshold: float = -1.0,
                 placeholder_expert: bool = False,
                 view_num: int = 1) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        # if k != 1 and k != 2:
        #     raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.placeholder_expert = placeholder_expert
        self.view_num = view_num
        if placeholder_expert:
            num_experts += 1
        self.wg = torch.nn.Linear(model_dim, num_experts * view_num, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = True
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        
        self.threshold = threshold

    def forward(self,
                input: torch.Tensor,
                used_token: torch.Tensor = None,
                use_tutel: bool = False, 
                in_logits: torch.Tensor = None, 
                now_training_process: float = None) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        if in_logits is None:
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
            input_fp32 = input.float()
            # input jittering
            if self.noisy_gate_policy == 'Jitter' and self.training:
                input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
            logits = self.wg(input_fp32)
        else:
            logits = in_logits

        if self.view_num > 1:
            logits = logits.reshape(logits.shape[0], -1, self.view_num)
            logits = torch.max(logits, dim=-1)[0].contiguous()

        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, use_tutel, placeholder_expert=self.placeholder_expert)

        else:
            # gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
            #                          self.min_capacity)

            if self.placeholder_expert:
                gate_output = plus_one_thresholdGating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.k, self.threshold)
            else:
                gate_output = new_thresholdGating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.k, self.threshold)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

        self.use_tutel = use_tutel and TUTEL_INSTALLED and gate.k == 1

        if self.use_tutel:
            logger.info('Using Tutel optimizations.')
        elif use_tutel and not TUTEL_INSTALLED:
            logger.warning("Tutel optimization requested but not installed. "
                           "Proceeding without Tutel.")
        elif use_tutel and TUTEL_INSTALLED and gate.k != 1:
            logger.warning("To enable Tutel optimization, use top-1 instead of top-2 gate. "
                           "Proceeding without Tutel.")

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        # if self.wall_clock_breakdown:
        self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        sequence_len = input[0].shape[0]
        d_model = input[0].shape[-1]

        reshaped_input = input[0].reshape(-1, d_model)
        
        ###### I want to use memory's key as gate // disabled ###############
        ################！！！！！！！！！！！！！！！！！！！！！！！！！
        logits = None
        
        ## average ffn keys 
        # these_local_keys = self.experts.get_key_weights().float()

        ## todo: all-to-all
        ## calculate logits
        # input_fp32 = reshaped_input.float()

        # logits = F.linear(input_fp32, these_local_keys)
        ################！！！！！！！！！！！！！！！！！！！！！！！！！


        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)

        if self.use_tutel:
            raise Exception("tutel honoka here!!!!!")
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts, self.gate_info  = self.gate(reshaped_input, input[1], in_logits=logits, now_training_process=kwargs.get('now_training_process', None))
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()

        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, dim=1)

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output, non_zero_ratio = self.experts(dispatched_input)
        self.gate_info["non_zero_ratio"] = non_zero_ratio

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if groups._get_expert_model_parallel_world_size() == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        a = combined_output.reshape(input[0].shape)

        # if self.wall_clock_breakdown:
        self.timers(MOE_TIMER).stop()
        self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        self.gate_info['gate_time'] = self.gate.gate_time
        self.gate_info['moe_time'] = self.time_moe

        return a
