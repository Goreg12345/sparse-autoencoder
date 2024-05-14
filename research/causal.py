import sys

sys.path.append('/tmp/pycharm_project_349')
sys.path.append('/tmp/pycharm_project_349/ioi_subspaces')

import torch
from mandala.all import *
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from autoscoring import run_with_cache
from ioi_subspaces.ioi_utils import PromptDataset, get_model, PromptDistribution, train_distribution, test_distribution, \
    full_distribution, MODEL_ID, MODELS
from ioi_subspaces.new_codebase import get_forced_hook, generate_prompts, FEATURE_SUBSETS, get_cf_prompts, \
    generate_name_samples
from ioi_subspaces.ioi_utils import Node

HEAD_CLASS_FIG = {
    'nm': 'Name Mover',
    'bnm': 'Backup Name Mover',
    'ind': 'Induction',
    'nnm': 'Negative Name Mover',
    'si': 'S-Inhibition',
    'dt': 'Duplicate Token',
    'pt': 'Previous Token',
}

COMPONENT_NAME_FIG = {
    'k': 'Key',
    'v': 'Value',
    'q': 'Query',
    'z': 'Attn Output',
}

CROSS_SECTION_FIG = {
    'ind+dt@z': 'Ind+DT out',
    'nm+bnm@q': '(B)NM q',
    'nm+bnm@qk': '(B)NM qk',
    'nm+bnm@z': '(B)NM out',
    'si@v': 'S-I v',
    'si@z': 'S-I out',
}

@op
def run_activation_patch(
    base_prompts: Any, # List[Prompt],
    cf_prompts: Any, # List[Prompt],
    nodes: List[Node],
    activations: List[Tensor],
    batch_size: int,
    model: HookedTransformer,
    return_predictions: bool = False,
) -> tuple[Tensor, tuple[Tensor, Tensor]] | tuple[Tensor, Tensor]:
    """
    Run a standard activation patch in a batched way
    """
    assert all([len(base_prompts) == v.shape[0] for v in activations])
    n = len(base_prompts)
    n_batches = (n + batch_size - 1) // batch_size
    base_logits_list = []
    cf_logits_list = []
    predictions_list = []
    for i in tqdm(range(n_batches)):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        prompts_batch = base_prompts[batch_indices]
        cf_batch = cf_prompts[batch_indices]
        base_dataset = PromptDataset(prompts_batch, model=model)
        cf_dataset = PromptDataset(cf_batch, model=model)
        hooks = [get_forced_hook(prompts=prompts_batch, node=node, A=act[batch_indices]) for node, act in zip(nodes, activations)]
        changed_logits = model.run_with_hooks(base_dataset.tokens, fwd_hooks=hooks)[:, -1, :]
        base_answer_logits = changed_logits.gather(dim=-1, index=base_dataset.answer_tokens.cuda())
        cf_answer_logits = changed_logits.gather(dim=-1, index=cf_dataset.answer_tokens.cuda())
        base_logits_list.append(base_answer_logits)
        cf_logits_list.append(cf_answer_logits)
        predictions = changed_logits.argmax(dim=-1)
        predictions_list.append(predictions)
    base_logits = torch.cat(base_logits_list, dim=0)
    cf_logits = torch.cat(cf_logits_list, dim=0)
    predictions = torch.cat(predictions_list, dim=0)
    if return_predictions:
        return base_logits, (cf_logits, predictions) #! lol, lmfaoooo
    else:
        return base_logits, cf_logits


@op
def extract_activations(distribution, circuit_nodes, feature_subsets):
    ### editing
    names = distribution.names
    N_NAMES = len(names)

    P_edit = generate_prompts(
        distribution=distribution,
        patterns=['ABB', 'BAB'],
        prompts_per_pattern=2500,
        random_seed=1,
    )

    batch_size = 100
    dataset = PromptDataset(prompts=P_edit, model=model)
    As_to_edit_dict = {node: [] for node in circuit_nodes}
    for i in range(0, len(P_edit), batch_size):
        batch = dataset[i:i + batch_size]
        A_to_edit = run_with_cache(llm=model, prompt_dataset=batch, nodes=circuit_nodes)
        for node, A in zip(circuit_nodes, A_to_edit):
            As_to_edit_dict[node].append(A)
    for node in circuit_nodes:
        As_to_edit_dict[node] = torch.cat(As_to_edit_dict[node], dim=0)

    N_EDIT = len(unwrap(P_edit))
    N_NAMES_EDIT_SOURCE = N_NAMES
    ### Compute counterfactual prompts

    def generate_name_samples(n_samples, names, random_seed: int = 0, exclude=None) -> Any:
        np.random.seed(random_seed)
        initial_choices = np.random.choice(names, n_samples, replace=True)
        if exclude is not None:
            for i, (name, exclude) in enumerate(zip(initial_choices, exclude)):
                if exclude is not None:
                    while name in exclude:
                        name = np.random.choice(names)
                    initial_choices[i] = name
        return initial_choices

    cf_prompts_dict = {}
    for feature_subset in feature_subsets:
        exclude = [p.names for p in P_edit]
        cf_prompts_dict[feature_subset] = get_cf_prompts(
            prompts=P_edit,
            features=feature_subset,
            # io_targets=[editing_source_distribution.names[0] for _ in range(N_EDIT)],
            # s_targets=[editing_source_distribution.names[1] for _ in
            # range(N_EDIT)],
            io_targets=generate_name_samples(N_EDIT, distribution.names[:N_NAMES_EDIT_SOURCE // 2], exclude=exclude),
            s_targets=generate_name_samples(N_EDIT, distribution.names[N_NAMES_EDIT_SOURCE // 2:], exclude=exclude),
        )
    ### Compute counterfactual activations
    As_counterfactual = {}
    for feature_subset, cf_prompts in tqdm(cf_prompts_dict.items()):
        As_counterfactual[feature_subset] = {n: [] for n in circuit_nodes}
        for i in range(0, len(P_edit), batch_size):
            batch = cf_prompts_dict[feature_subset][i:i + batch_size]
            batch = PromptDataset(prompts=batch, model=model)
            As_counterfactual_batch = run_with_cache(llm=model, prompt_dataset=batch, nodes=circuit_nodes)
            for node, A in zip(circuit_nodes, As_counterfactual_batch):
                As_counterfactual[feature_subset][node].append(A)
        for node in circuit_nodes:
            As_counterfactual[feature_subset][node] = torch.cat(As_counterfactual[feature_subset][node], dim=0)
        torch.cuda.empty_cache()
    return As_to_edit_dict, As_counterfactual
#%%


if __name__ == '__main__':
    import sys
    sys.path.append('/tmp/pycharm_project_349')
    # input to the function
    # the circuit
    # -> then load the SAEs for the circuit
    # model
    model = get_model(config='webtext')
    MODELS[MODEL_ID] = model
    distribution = full_distribution
    storage = Storage(db_path='autoencoders.db', spillover_dir='spillover_saes/')

    feature_subsets = FEATURE_SUBSETS

    circuit_nodes = [Node(component_name='z', layer=9, head=6, seq_pos='end'), Node(component_name='z', layer=9, head=9, seq_pos='end'), Node(component_name='z', layer=10, head=0, seq_pos='end')]
    actvs = extract_activations(distribution, circuit_nodes, feature_subsets)
