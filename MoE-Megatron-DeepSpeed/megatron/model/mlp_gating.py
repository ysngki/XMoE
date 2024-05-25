from typing import Callable, Dict, Tuple, Optional
from deepspeed import comm as dist

import torch
from torch import Tensor
import torch.nn.functional as F

import copy
import math

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
def _top_idx(source, k):
	return torch.topk(source, k=k, dim=0)[1]


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


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


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
	gumbel = gumbel_map.get(device)
	if gumbel is None:
		one = torch.tensor(1.0, device=device)
		zero = torch.tensor(0.0, device=device)
		gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
		gumbel_map[device] = gumbel
	return gumbel(shape)


class Experts(torch.nn.Module):
	def __init__(self, expert, num_local_experts=1):
		super(Experts, self).__init__()

		self.yyh_local_experts = torch.nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])

	def forward(self, inputs, inputs_weight, top_idx):
		# inputs: (s, m), inputs_weight: (s, e)
		expert_output = torch.zeros_like(inputs)
		out_non_zero_ratio = None
		for e_idx, expert in enumerate(self.yyh_local_experts):
			token_idx = top_idx[:, e_idx]  # (capacity)
			these_tokens = inputs[token_idx]  # (capacity, dim)

			out = expert(these_tokens)

			if type(out) is tuple:
				if out_non_zero_ratio is None:
					out_non_zero_ratio = out[2]
				else:
					out_non_zero_ratio += out[2]

				out = out[0]  # Ignore the bias term for now

			expert_output[token_idx] += out * inputs_weight[:, e_idx][token_idx].unsqueeze(-1).type_as(inputs)

		return expert_output, out_non_zero_ratio / len(self.yyh_local_experts)


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
	top1_p, _ = torch.max(gates, dim=1)

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

	all_valid_chosen_num = mask1.sum()
	avg_valid_chosen_num = all_valid_chosen_num / mask1.shape[0]

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
	# locations1_s = torch.sum(locations1 * mask1, dim=1)

	# Normalize gate probabilities
	mask1_float = mask1.float()
	gates = gates * mask1_float

	# locations1_sc = _one_hot_to_float(locations1_s, capacity)
	# combine_weights = einsum("se,sc->sec", gates, locations1_sc)
	combine_weights = gates

	# dispatch_mask = combine_weights.bool()
	dispatch_mask = None

	token_chosen_num = mask1.sum(dim=-1)
	token_not_full_ratio = (token_chosen_num < 1).sum() / token_chosen_num.shape[0]

	gate_info = {
		"expert_not_full_ratio": expert_not_full_ratio,
		"chosen_num": avg_valid_chosen_num,
		"token_not_full_ratio": token_not_full_ratio,
		"top1_p": top1_p
	}

	if placeholder_expert:
		skip_ratio = (indices1_s == 0).sum() / indices1_s.shape[0]

		gate_info["skip_ratio"] = skip_ratio

	return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info, top_idx


def topkgating(logits: Tensor, capacity_factor: float, min_capacity: int, in_k: int) -> Tuple[
	Tensor, Tensor, Tensor, Tensor]:
	# everything is in fp32 in this function
	gates = F.softmax(logits, dim=1)
	top1_p, _ = torch.max(gates, dim=1)

	capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

	# Create a mask for 1st's expert per token
	indices1_s = torch.argmax(gates, dim=1)
	num_experts = int(gates.shape[1])
	mask1 = F.one_hot(indices1_s, num_classes=num_experts)

	# gating decisions (no use)
	exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

	# select remaining (k-1) experts
	logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
	# Replace top-expert with min value
	logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))

	if in_k > 1:
		_, gates_indices = torch.topk(logits_except1, dim=-1, k=in_k-1, largest=True, sorted=True)
		whole_chosen_indices = torch.cat([indices1_s.unsqueeze(-1), gates_indices], dim=-1) # (token num, in_k)
	else:
		whole_chosen_indices = indices1_s.unsqueeze(-1)


	scatter_importance = torch.arange(in_k, 0, -1, device=whole_chosen_indices.device).expand(
		whole_chosen_indices.shape)  # (token num, in_k)
	tensor_all_mask = torch.zeros((whole_chosen_indices.shape[0], num_experts), device=whole_chosen_indices.device,
								  dtype=scatter_importance.dtype).scatter_(1, whole_chosen_indices,
																			 scatter_importance) # (token num, expert_num)
	token_num, expert_num = tensor_all_mask.shape

	expert_received_num = (tensor_all_mask > 0).sum(dim=0)
	receive_ratio = expert_received_num * 100 / token_num

	top_idx = _top_idx(tensor_all_mask, capacity)  # (capacity, expert num)
	new_mask1 = tensor_all_mask * torch.zeros_like(tensor_all_mask).scatter_(0, top_idx, 1)
	tensor_all_mask = (new_mask1 > 0).int()
	# -------------------------------------------
	expert_actual_received_num = tensor_all_mask.sum(dim=0)
	expert_not_full_ratio = (expert_actual_received_num < capacity).sum() / expert_num

	# Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
	me = torch.mean(gates, dim=0)
	ce = torch.mean(mask1.float(), dim=0)
	l_aux = torch.sum(me * ce) * num_experts
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????

	token_chosen_num = tensor_all_mask.sum(dim=-1)  # (token_num)
	token_not_full_ratio = (token_chosen_num < in_k).sum() / token_num

	all_valid_chosen_num = token_chosen_num.sum()
	avg_valid_chosen_num = all_valid_chosen_num / token_num

	tensor_all_mask_float = tensor_all_mask.float()
	tensor_all_gates = gates * tensor_all_mask_float  # (s, e)
	combine_weights = tensor_all_gates
	dispatch_mask = None  # no used actually

	gate_info = {
		"top1_p": top1_p,
		"chosen_num": avg_valid_chosen_num,
		"token_not_full_ratio": token_not_full_ratio,
		"expert_not_full_ratio": expert_not_full_ratio,
		"receive_ratio": receive_ratio
	}
	return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info, top_idx


def baselayer_gating(logits: Tensor, capacity_factor: float, min_capacity: int, in_k: int, training_flag):
	capacity = _capacity(logits, torch.tensor(capacity_factor), torch.tensor(min_capacity))

	# just for logging
	gates = F.softmax(logits, dim=1)
	top1_p, _ = torch.max(gates, dim=1)
	####

	# use top-1 gating for evaluation
	if not training_flag:
		indices1_s = torch.argmax(logits, dim=1)
		num_experts = int(logits.shape[1])
		mask1 = F.one_hot(indices1_s, num_classes=num_experts)

		top_idx = _top_idx(mask1, capacity)
		mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
		mask1_float = mask1.float()
		combine_weights = torch.sigmoid(logits) * mask1_float

		gate_info = {}

		return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), combine_weights, None, None, gate_info, top_idx

	# use linear assignment for training
	from fairseq import libbase

	ok = logits.isfinite()
	if not ok.all():
		# NaNs here can break the assignment algorithm
		logits[~ok] = logits[ok].min()

	assigment = libbase.balanced_assignment(logits.detach()) # (token_num)

	assigment = assigment.reshape(logits.shape[1], logits.shape[0] // logits.shape[1]) # (expert_num, token_num / expert_num)
	top_idx = assigment.t()[:capacity, :] # (capacity, expert num)

	tensor_all_mask = torch.zeros_like(logits).scatter_(0, top_idx, 1)
	combine_weights = torch.sigmoid(logits) * tensor_all_mask

	gate_info = {
		"top1_p": top1_p,
	}

	return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), combine_weights, None, None, gate_info, top_idx


def main_thresholdGating(logits: Tensor, capacity_factor: float, min_capacity: int, k: int, threshold: float) -> Tuple[
	Tensor, Tensor, Tensor, Tensor]:
	# everything is in fp32 in this function
	gates = F.softmax(logits, dim=1)

	top1_p, _ = torch.max(gates, dim=1)

	capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

	# Create a mask for 1st's expert per token
	indices1_s = torch.argmax(gates, dim=1)
	num_experts = int(gates.shape[1])
	mask1 = F.one_hot(indices1_s, num_classes=num_experts)

	# gating decisions (no use)
	exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

	explore_top_k_num = num_experts
	gates_sorted, gates_indices = torch.topk(gates, dim=-1, k=explore_top_k_num, largest=True, sorted=True)

	cum_sorted_gates = torch.cumsum(gates_sorted, dim=-1)
	chosen_flag = (cum_sorted_gates - gates_sorted) < threshold  # (token num, explore_top_k_num)
	chosen_flag[:, 0] = True  # at least select one expert
	whole_chosen_indices = chosen_flag * (
			gates_indices + 1)  # (token num, explore_top_k_num) \in (0, expert_num + 1), 0 means not choose

	# get masks and capacity locations
	explore_top_k_num = whole_chosen_indices.sum(dim=0).ne(0).sum()
	whole_chosen_indices = whole_chosen_indices[:, :explore_top_k_num]  # (token num, explore_top_k_num)
	prob_importance = gates_sorted[:, :explore_top_k_num]

	each_token_want_num = (whole_chosen_indices > 0).sum(dim=1)
	avg_want_num = each_token_want_num.sum() / each_token_want_num.shape[0]

	scatter_importance = torch.arange(explore_top_k_num, 0, -1, device=whole_chosen_indices.device).expand(
		whole_chosen_indices.shape)  + prob_importance # (token num, explore_top_k_num)
	tensor_all_mask = torch.zeros((whole_chosen_indices.shape[0], num_experts + 1), device=whole_chosen_indices.device,
								  dtype=scatter_importance.dtype).scatter_(1, whole_chosen_indices,
																			 scatter_importance)[:,
					  1:]  # (token num, expert_num)
	token_num, expert_num = tensor_all_mask.shape

	# random token selection (ignore position)
	expert_received_num = (tensor_all_mask > 0).sum(dim=0)
	receive_ratio = expert_received_num * 100 / token_num

	top_idx = _top_idx(tensor_all_mask, capacity)  # (capacity, expert num)
	new_mask1 = tensor_all_mask * torch.zeros_like(tensor_all_mask).scatter_(0, top_idx, 1)
	tensor_all_mask = (new_mask1 > 0).int()
	# -------------------------------------------

	# tensor_all_locations = torch.cumsum(tensor_all_mask, dim=0) - 1 # (token_num, expert_num)

	expert_received_num = tensor_all_mask.sum(dim=0)
	expert_not_full_ratio = (expert_received_num < capacity).sum() / expert_num

	# tensor_all_locations = tensor_all_locations * tensor_all_mask # (token_num, expert_num)

	#######################################################################
	### my simple loss
	me = torch.mean(logits, dim=0)
	l_aux = -me / num_experts
	###

	# Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
	me = torch.mean(gates, dim=0)

	ce = torch.mean(mask1.float(), dim=0)

	l_aux = torch.sum(me * ce) * num_experts
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
	#######################################################################

	token_chosen_num = tensor_all_mask.sum(dim=-1)  # (token_num)
	token_not_full_ratio = (token_chosen_num < each_token_want_num).sum() / token_num

	all_valid_chosen_num = token_chosen_num.sum()
	avg_valid_chosen_num = all_valid_chosen_num / token_num

	tensor_all_mask_float = tensor_all_mask.float()
	tensor_all_gates = gates * tensor_all_mask_float  # (s, e)

	combine_weights = tensor_all_gates
	dispatch_mask = None  # no used actually

	gate_info = {
		"top1_p": top1_p,
		"chosen_num": avg_valid_chosen_num,
		"token_not_full_ratio": token_not_full_ratio,
		"expert_not_full_ratio": expert_not_full_ratio,
		"want_num": avg_want_num,
		"receive_ratio": receive_ratio
	}
	return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info, top_idx


class TopKGate(torch.nn.Module):
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
				 view_num: int = 1,
				 num_local_experts: int = 0,
				 scale_moe: bool = False) -> None:
		super().__init__()

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
		self.gate_time = 0.0
		self.drop_tokens = drop_tokens
		self.use_rts = use_rts

		self.threshold = threshold
		self.num_local_experts = num_local_experts

		self.scale_moe = scale_moe
		if self.scale_moe:
			self.down_proj_1 = torch.nn.Linear(model_dim, model_dim // k).float()
			# self.down_proj_2 = torch.nn.Linear(model_dim, model_dim // k).float()

			# self.up_proj_1 = torch.nn.Linear(model_dim // k, model_dim).float()
			self.up_proj_2 = torch.nn.Linear(model_dim // k, model_dim).float()

	def forward(self,
				input: torch.Tensor,
				used_token: torch.Tensor = None,
				use_tutel: bool = False,
				in_logits: torch.Tensor = None,
				now_training_process: float = None,
				gating_function=None,
				use_base_layer: bool = False,
				use_topk: bool =False,
				use_threshold: bool =False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

		assert self.training == input.requires_grad, "Traning Flag Wrong in Router, YYH!"

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

			# aggregate scores to get a final score
			temp_warm_frac = 0.1
			temp_process = min(now_training_process / temp_warm_frac, 1.0)
			this_temperature = (math.cos(temp_process * math.pi) + 1) * 10.0 + 0.1

			logits_weight = torch.nn.functional.softmax(logits / this_temperature, dim=-1).detach()
			logits = (logits_weight * logits).sum(dim=-1)

		if use_base_layer:
			gate_output = baselayer_gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
									 self.min_capacity, self.k, self.training)
		elif use_topk:
			gate_output = topkgating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
									 self.min_capacity, self.k)

		elif use_threshold:
			if self.placeholder_expert:
				raise Exception("Not supported yet!")
				gate_output = plus_one_thresholdGating(logits,
													   self.capacity_factor if self.training else self.eval_capacity_factor,
													   self.min_capacity, self.k, self.threshold)
			else:
				gate_output = main_thresholdGating(logits,
												   self.capacity_factor if self.training else self.eval_capacity_factor,
												   self.min_capacity, self.k, self.threshold)
		else:
			raise Exception("I should designate the router to use")

		return gate_output


class HashRouter(torch.nn.Module):
	def __init__(self, num_experts, voc_size, capacity_factor, eval_capacity_factor, min_capacity):
		super().__init__()
		self.hash_bin_map = torch.nn.Parameter(torch.LongTensor(voc_size).fill_(0), requires_grad=False)

		import random
		for i in range(voc_size):
			self.hash_bin_map[i] = random.randrange(0, num_experts)

		self.num_experts = num_experts
		self.capacity_factor = torch.tensor(capacity_factor)
		self.eval_capacity_factor = torch.tensor(eval_capacity_factor)
		self.min_capacity = torch.tensor(min_capacity)
	
	def forward(self, input_ids):
		now_capacity_factor = self.capacity_factor if self.training else self.eval_capacity_factor
		capacity = torch.ceil((len(input_ids) / self.num_experts) * now_capacity_factor).to(torch.int64)
		if capacity < self.min_capacity:
			capacity = self.min_capacity.to(torch.int64)

		indices1_s = self.hash_bin_map[input_ids]
		mask1 = F.one_hot(indices1_s, num_classes=self.num_experts)

		top_idx = _top_idx(mask1, capacity)
		mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)

		all_valid_chosen_num = mask1.sum()
		avg_valid_chosen_num = all_valid_chosen_num / mask1.shape[0]

		combine_weights = mask1.float()

		gate_info = {
			"chosen_num": avg_valid_chosen_num,
		}

		return combine_weights, gate_info, top_idx


uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


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
