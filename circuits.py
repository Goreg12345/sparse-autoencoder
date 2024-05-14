import collections
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import re
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Any, Literal, Optional, Union, Tuple, Callable, Sequence, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset, BatchSampler, SequentialSampler, Sampler
from transformer_lens import HookedTransformer, ActivationCache

import sys
sys.path.append('/tmp/pycharm_project_349')
sys.path.append('/tmp/pycharm_project_349/ioi_subspaces')
from ioi_subspaces.ioi_utils import PromptDistribution, train_distribution, Prompt, batched


def generate_prompts(distribution: PromptDistribution, patterns: List[str],
                     prompts_per_pattern: int, model
                     ) -> Any:
    parts = [[distribution.sample_one(pattern=pattern, model=model)
              for _ in range(prompts_per_pattern)] for pattern in patterns]
    prompts = [p for part in parts for p in part]
    return prompts


class PromptDataset(Dataset):
    def __init__(self, prompts: List[Prompt], model: HookedTransformer):
        assert len(prompts) > 0
        self.prompts: List[Prompt] = prompts
        self.model = model
        ls = self.lengths
        if not all(x == ls[0] for x in ls):
            raise ValueError("Prompts must all have the same length")

    def __getitem__(self, idx: Union[int, Sequence, slice]) -> "PromptDataset":
        if isinstance(idx, int):
            prompts = [self.prompts[idx]]
        else:
            prompts = self.prompts[idx]
            if isinstance(prompts, Prompt):
                prompts = [prompts]
        assert all(isinstance(x, Prompt) for x in prompts)
        return PromptDataset(prompts=prompts, model=self.model)

    def __len__(self) -> int:
        return len(self.prompts)

    def __repr__(self) -> str:
        return f"{[x for x in self.prompts]}"

    def __rich_repr__(self):
        for x in self.prompts:
            yield x.sentence

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(
            prompts=list(self.prompts) + list(other.prompts), model=self.model
        )

    @property
    def lengths(self) -> List[int]:
        return [self.model.to_tokens(x.sentence).shape[1] for x in self.prompts]

    @property
    def io_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.io_name}") for x in self.prompts]
        )

    @property
    def s_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.s_name}") for x in self.prompts]
        )

    @property
    def tokens(self) -> Tensor:
        return self.model.to_tokens([x.sentence for x in self.prompts])
    @property
    def labels(self) -> Tensor:
        return self.io_tokens

    @property
    def answer_positions(self) -> Tensor:
        return torch.tensor([-1 for _ in self.prompts])

    @property
    def incorr_labels(self) -> Tensor:
        return self.s_tokens


class CircuitComponent(ABC):
    def get_value(self, cache: ActivationCache,
                  prompts: Optional[List[Prompt]] = None
                  ) -> Tensor:
        raise NotImplementedError

    @property
    def names_filter(self) -> Callable:
        raise NotImplementedError

    @property
    def stop_at_layer(self) -> int:
        raise NotImplementedError

    @property
    def displayname(self) -> str:
        raise NotImplementedError

    def get_hook_fn(self, f: Callable, prompts: Optional[List[Prompt]]) -> List[Tuple[Union[str, Callable], Callable]]:
        raise NotImplementedError

    def get_value_as_dict(self, cache, prompts):
        raise NotImplementedError


@dataclass
class Node(CircuitComponent):
    def __init__(
            self,
            component_name: Literal[
                "z",
                "attn_out",
                "pre",
                "post",
                "mlp_out",
                "resid_pre",
                "resid_post",
                "resid_mid",
                "q",
                "k",
                "v",
                "pattern",
                "attn_scores",
                "result",
                "q_input",
                "k_input",
                "v_input",
                'scale_ln1',
                'scale_ln2',
                'scale_final',
                "ln_final",
            ],
            layer: Optional[int] = None,
            head: Optional[int] = None,
            neuron: Optional[int] = None,
            seq_pos: Optional[Union[int, str]] = None,  # string used for semantic indexing
    ):
        assert isinstance(component_name, str)
        self.component_name = component_name
        if layer is not None:
            assert isinstance(layer, int)
        self.layer = layer
        if head is not None:
            assert isinstance(head, int)
        self.head = head
        if neuron is not None:
            assert isinstance(neuron, int)
        self.neuron = neuron
        if seq_pos is not None:
            assert isinstance(seq_pos, (int, str))
        self.seq_pos = seq_pos

    def get_seq_pos_from_prompt(self, prompt: Prompt):
        """
        Return a new node with the seq_pos resolved to an integer.
        """
        if isinstance(self.seq_pos, str):
            return Node(
                component_name=self.component_name,
                layer=self.layer,
                head=self.head,
                neuron=self.neuron,
                seq_pos=prompt.semantic_pos[self.seq_pos],
            )
        else:
            return self

    @property
    def actv_name(self):
        """
        Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback
        loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

        Returns a name corresponding to an activation point in a TransformerLens model.

        Args:
             name (str): Takes in the name of the activation. This can be used to specify any activation name by itself.
             The code assumes the first sequence of digits passed to it (if any) is the layer number, and anything after
             that is the layer type.

             Given only a word and number, it leaves layer_type as is.
             Given only a word, it leaves layer and layer_type as is.

             Examples:
                 get_act_name('embed') = get_act_name('embed', None, None)
                 get_act_name('k6') = get_act_name('k', 6, None)
                 get_act_name('scale4ln1') = get_act_name('scale', 4, 'ln1')

             layer (int, optional): Takes in the layer number. Used for activations that appear in every block.

             layer_type (string, optional): Used to distinguish between activations that appear multiple times in one block.

        Full Examples:

        get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
        get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
        get_act_name('embed')=='hook_embed'
        get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
        get_act_name('k6')=='blocks.6.attn.hook_k'
        get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
        get_act_name('pre5')=='blocks.5.mlp.hook_pre'
        """
        if (
                ("." in self.component_name or self.component_name.startswith("hook_"))
                and self.layer is None
        ):
            # If this was called on a full name, just return it
            return self.component_name
        match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", self.component_name)
        if match is not None:
            self.component_name, self.layer, layer_type = match.groups(0)

        act_name_alias = {
            "attn": "pattern",
            "attn_logits": "attn_scores",
            "key": "k",
            "query": "q",
            "value": "v",
            "mlp_pre": "pre",
            "mlp_mid": "mid",
            "mlp_post": "post",
        }

        layer_norm_names = ["scale", "normalized"]

        if self.component_name in act_name_alias:
            self.component_name = act_name_alias[self.component_name]

        full_act_name = ""
        if self.layer is not None:
            full_act_name += f"blocks.{self.layer}."
        if self.component_name in [
            "k",
            "v",
            "q",
            "z",
            "rot_k",
            "rot_q",
            "result",
            "pattern",
            "attn_scores",
        ]:
            layer_type = "attn"
        elif self.component_name in ["pre", "post", "mid", "pre_linear"]:
            layer_type = "mlp"
        else:
            layer_type = None

        if layer_type:
            full_act_name += f"{layer_type}."
        full_act_name += f"hook_{self.component_name}"

        if self.component_name in layer_norm_names and self.layer is None:
            full_act_name = f"ln_final.{full_act_name}"
        return full_act_name

    @property
    def shape_type(self) -> List[str]:
        """
        List of the meaning of each dimension of the full activation for this
        node (i.e., what you'd get if you did `cache[self.activation_name]`).

        This is just for reference
        """
        if self.component_name in [
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q_input",
            "k_input",
            "v_input",
        ]:
            return ["batch", "seq", "d_model"]
        elif self.component_name == 'pattern':
            return ["batch", "head", "query_pos", "key_pos"]
        elif self.component_name in ["q", "k", "v", "z"]:
            return ["batch", "seq", "head", "d_head"]
        elif self.component_name in ["result"]:
            return ["batch", "seq", "head", "d_model"]
        elif self.component_name == 'scale':
            return ['batch', 'seq']
        elif self.component_name == 'post':
            return ['batch', 'seq', 'd_mlp']
        else:
            raise NotImplementedError

    def idx(self, prompts: Optional[List[Prompt]] = None) -> Tuple[Union[int, slice, Tensor, None], ...]:
        """
        Index into the full activation to restrict to layer / head / neuron /
        seq_pos
        """
        if isinstance(self.seq_pos, str):
            assert prompts is not None
            seq_pos_idx = torch.Tensor([p.semantic_pos[self.seq_pos] for p in prompts]).long()
            batch_idx = torch.arange(len(prompts)).long()
        elif isinstance(self.seq_pos, int):
            seq_pos_idx = self.seq_pos
            batch_idx = slice(None)
        elif self.seq_pos is None:
            seq_pos_idx = slice(None)
            batch_idx = slice(None)
        else:
            raise NotImplementedError

        if self.neuron is not None:
            raise NotImplementedError

        elif self.component_name in ['pattern', 'attn_scores']:
            assert self.head is not None
            return tuple([slice(None), self.head, slice(None), slice(None)])
        elif self.component_name in ["q", "k", "v", "z", "result"]:
            assert self.head is not None, "head must be specified for this component"
            return tuple([batch_idx, seq_pos_idx, self.head, slice(None)])
        elif self.component_name == 'scale':
            return tuple([slice(None), slice(None)])
        elif self.component_name == 'post':
            return tuple([batch_idx, seq_pos_idx, slice(None)])
        else:
            return tuple([batch_idx, seq_pos_idx, slice(None)])

    @property
    def names_filter(self) -> Callable:
        return lambda x: x in [self.actv_name]

    @property
    def needs_head_results(self) -> bool:
        return self.component_name in ['result']

    def get_value(self, cache: ActivationCache,
                  prompts: Optional[List[Prompt]] = None
                  ) -> Tensor:
        return cache[self.actv_name][self.idx(prompts=prompts)]

    def get_value_as_dict(self, cache: ActivationCache,
                            prompts: Optional[List[Prompt]] = None
                            ) -> OrderedDict[str, Tensor]:
            return OrderedDict([(self.displayname, self.get_value(cache, prompts))])

    @property
    def stop_at_layer(self) -> int:
        if self.layer:
            return self.layer + 1
        else:
            return -1

    @property
    def displayname(self) -> str:
        if self.component_name in ('q', 'k', 'v', 'z'):
            return f'{self.component_name}@L{self.layer}H{self.head}@{self.seq_pos}'
        else:
            raise NotImplementedError

    def get_hook_fn(self, f: Callable, prompts) -> List[Tuple[Union[str, Callable], Callable]]:
        def hook_fn(actv, hook):
            idx = self.idx(prompts=prompts)
            node_actv = actv[idx]
            node_actv = f(node_actv)
            actv[idx] = node_actv
            return actv
        return [(self.names_filter, hook_fn)]


class Circuit(CircuitComponent):
    def __init__(self, subcircuits: List['CircuitComponent'], displayname: Optional[str] = 'Circuit'):
        self.subcircuits = subcircuits
        self._displayname = displayname

    @property
    def names_filter(self) -> Callable:
        return lambda x: any(subcircuit.names_filter(x) for subcircuit in self.subcircuits)

    @property
    def stop_at_layer(self) -> int:
        if any(subcircuit.stop_at_layer == -1 for subcircuit in self.subcircuits):
            return -1
        else:
            return max(subcircuit.stop_at_layer for subcircuit in self.subcircuits)

    def get_value(self, cache: ActivationCache,
                  prompts: Optional[List[Prompt]] = None
                  ) -> Tensor:
        return torch.cat([subcircuit.get_value(cache, prompts) for subcircuit in self.subcircuits], dim=-1)

    def get_value_as_dict(self, cache: ActivationCache,
                          prompts: Optional[List[Prompt]] = None
                          ) -> OrderedDict[str, Tensor]:
        return OrderedDict([(subcircuit.displayname, subcircuit.get_value_as_dict(cache, prompts)) for subcircuit in self.subcircuits])

    @property
    def displayname(self) -> str:
        return self._displayname

    def get_hook_fn(self, f: Callable, prompts) -> List[Tuple[Union[str, Callable], Callable]]:
        hooks = []
        for subcircuit in self.subcircuits:
            hooks.extend(subcircuit.get_hook_fn(f, prompts))
        return hooks


@dataclass
class Result:
    prompts: PromptDataset
    logits: Tensor
    loss: Tensor
    device: str = 'cuda'

    def accuracy(self) -> float:
        answer_positions = self.prompts.answer_positions.to(self.device)
        labels = self.prompts.labels.to(self.device)
        return (self.logits[torch.arange(len(self.prompts), device=self.device),
                answer_positions, :].argmax(dim=1) == labels).float().mean().item()

    def logit_diff(self) -> Tensor:
        corr_logits = self.logits[torch.arange(len(self.logits)), self.prompts.answer_positions, self.prompts.labels]
        incorr_logits = self.logits[torch.arange(len(self.logits)), self.prompts.answer_positions, self.prompts.incorr_labels]
        return corr_logits - incorr_logits


class Brain:
    def __init__(self, llm: HookedTransformer, fwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = [],
                    bwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = []):
        self.llm = llm

class CircuitActivations:
    # holds activations for a circuit, similar to ActivationCache but for every (nested) component in a circuit rather than just complete components
    def __init__(self, activations: OrderedDict[str, Tensor]):
        self.activations = activations

    def patch_in(self,):
        print('Im patching')

#    def ablate

class GeneratorTask:
    def __init__(self, circuit: CircuitComponent, llm: HookedTransformer, prompts: torch.utils.data.DataLoader,
                 device: str = 'cuda',
                 fwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = [],
                 bwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = []):
        self.circuit = circuit
        self.llm = llm
        self.prompts = prompts
        self.device = device
        # we don't wanna register the hooks directly in the LLM, because it might be used for other operations
        # and we don't want to copy the LLM because it's large
        # instead, we save the hooks in the Task object and use them when we need to
        self.fwd_hooks = fwd_hooks
        self.bwd_hooks = bwd_hooks

    @torch.no_grad()
    def forward(self) -> Iterable[Result]:
        llm.to(self.device)
        for batch in self.prompts:
            tokens = batch.tokens.to(self.device)
            with llm.hooks(fwd_hooks=self.fwd_hooks, bwd_hooks=self.bwd_hooks):
                logits, loss = self.llm(tokens, return_type='both')
            yield Result(prompts=batch, logits=logits, loss=loss, device=self.device)

    @torch.no_grad()
    def get_activations(self) -> Iterable[CircuitActivations]:
        llm.to(self.device)
        for batch in self.prompts:
            tokens = batch.tokens.to(self.device)
            with llm.hooks(fwd_hooks=self.fwd_hooks, bwd_hooks=self.bwd_hooks):
                output, cache = self.llm.run_with_cache(tokens, return_type='both')
            yield CircuitActivations(self.circuit.get_value_as_dict(cache, batch.prompts))


class Task:
    def __init__(self, circuit: CircuitComponent, llm: HookedTransformer, prompts: List[Prompt], device: str = 'cuda',
                 fwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = [],
                 bwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = []):
        self.circuit = circuit
        self.llm = llm
        self.dataset = PromptDataset(prompts=prompts, model=llm)
        self.prompts = prompts
        self.device = device
        # we don't wanna register the hooks directly in the LLM, because it might be used for other operations
        # and we don't want to copy the LLM because it's large
        # instead, we save the hooks in the Task object and use them when we need to
        self.fwd_hooks = fwd_hooks
        self.bwd_hooks = bwd_hooks

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, batch_size: int = 200) -> Result:
        llm.to(self.device)
        with torch.no_grad():
            all_logits = []
            all_loss = []
            for i in range(0, len(self.dataset), batch_size):
                batch = self.dataset[i:i + batch_size]
                tokens = batch.tokens.to(self.device)
                with llm.hooks(fwd_hooks=self.fwd_hooks, bwd_hooks=self.bwd_hooks):
                    logits, loss = self.llm(tokens, return_type='both')
                all_logits.append(logits)
                all_loss.append(loss)
            logits = torch.cat(all_logits, dim=0)
            loss = torch.tensor(all_loss, device=self.device).mean()
        return Result(prompts=self.dataset, logits=logits, loss=loss, device=self.device)

    def run_with_cache(self, batch_size: int = 200) -> Tuple[Result, ActivationCache]:
        llm.to(self.device)
        with torch.no_grad():
            all_caches = []
            for i in range(0, len(self.dataset), batch_size):
                batch = self.dataset[i:i + batch_size]
                tokens = batch.tokens.to(self.device)
                with llm.hooks(fwd_hooks=self.fwd_hooks, bwd_hooks=self.bwd_hooks):
                    output, cache = self.llm.run_with_cache(tokens, return_type='both',
                                                                  stop_at_layer=self.circuit.stop_at_layer,
                                                                  names_filter=self.circuit.names_filter)
                all_caches.append(cache)
                return cache

    def ablate(self, method: Union[Literal['zero', 'mean'], Callable] = 'zero') -> "Task":
        """
        Ablate the activations in the circuit
        """
        if method == 'zero':
            method = lambda x: torch.zeros_like(x)
        elif method == 'mean':
            method = lambda x: x.mean(dim=1, keepdim=True)
        else:
            assert callable(method)
        fwd_hooks = self.circuit.get_hook_fn(method, self.prompts)
        fwd_hooks = self.fwd_hooks + fwd_hooks
        return Task(circuit=self.circuit, llm=self.llm, prompts=self.dataset.prompts, fwd_hooks=fwd_hooks, bwd_hooks=self.bwd_hooks)

    def patch(self, cache: ActivationCache) -> "Task":
        ...
        # get cache
        # batch?
        # prepare hook functions that
        hooks = self.circuit.get_hook_fn(lambda x: cache[x], self.prompts)
    # mmh but in reality this function should probably take activations in and not prompts
    # or an activations generator
    # so basically, I'd need to make new hook functions for every new batch
    # and basically, I have two different types of modi:
    #    1. do every single step on the entire dataset, aggregate results, and essentially have a data pipeline
    #    2. do every single step on a batch and have some kind of generator that generates the activations everywhere
    # but if I get a generator, does it have the same batch size?
    # the dataset should be a dataloader which is a generator
    # then, the whole shit will just follow and I never need to specify batch size
    # mmh but what if I have a second dataset that I wanna patch?

    def decompile(self, saes):
        ...


if __name__ == '__main__':
    llm = HookedTransformer.from_pretrained('gpt2-small')
    llm.eval()
    llm.requires_grad_(False)

    distribution = PromptDistribution(
        prefix_len=2,
        names=train_distribution.names,
        objects=train_distribution.objects,
        places=train_distribution.places,
        templates=train_distribution.templates,
        prefixes=train_distribution.prefixes
    )
    prompts = generate_prompts(distribution, patterns=['ABB', 'BAB'], prompts_per_pattern=100, model=llm)
    z_nm = Circuit([
        Node('z', layer=9, head=9, seq_pos='end'),
        Node('z', layer=9, head=6, seq_pos='end'),
        Node('z', layer=10, head=0, seq_pos='end')
    ], displayname='z@nm')
    q_nm = Circuit([
        Node('q', layer=9, head=9, seq_pos='end'),
        Node('q', layer=9, head=6, seq_pos='end'),
        Node('q', layer=10, head=0, seq_pos='end')
    ], displayname='q@nm')
    ioi = Circuit([z_nm, q_nm], displayname='IOI')

    def collate_fn(batch: List[Prompt]):
        return PromptDataset(prompts=batch, model=llm)
    prompts = torch.utils.data.DataLoader(prompts, collate_fn=collate_fn, batch_size=100)

    task = GeneratorTask(ioi, llm, prompts)
    accs = []
    for c in task.get_activations():
        c.patch_in()
    for result in task.forward():
        accs.append(result.accuracy())
    print(torch.tensor(accs).mean())
    cache = task.run_with_cache()
    circuit_actvs = task.circuit.get_value_as_dict(cache, prompts)
    print(result.accuracy())
    print(result.logit_diff())
    print(prompts)
    atask = task.ablate('zero')
    result = atask()
    print(result.accuracy())
    print(result.logit_diff())
    print(prompts)
    ...