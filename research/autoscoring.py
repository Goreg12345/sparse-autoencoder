from typing import Any, List, Union

from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
import torch
from activation_buffer import Buffer
from text_dataset import TextDataset
import text_dataset
from torch.utils.data import DataLoader


def get_f_score(precision, recall, beta=1):  # beta changes the importance of precision vs recall
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)


def run_with_cache(
    llm,
    prompt_dataset: Any,
    nodes: Union[Any, List],
    return_logits: bool = False,
    offload_to_cpu: bool = False,
    clear_cache: bool = False,
    **kwargs,
) -> List[Tensor]:
    """
    Run the model on the given prompts, and return the activations for the
    given nodes.
    """
    unwrap_list = True if not isinstance(nodes, list) else False
    if not isinstance(nodes, list):
        nodes = [nodes]
    names_filter = lambda x: any(node.names_filter(x) for node in nodes)
    tokens = prompt_dataset.tokens
    tokens = tokens.to(llm.W_E.device)
    logits, cache = llm.run_with_cache(tokens, names_filter=names_filter, **kwargs)
    # model.reset_hooks() ---> this is potentially confusing
    # return {node: node.get_value(cache, prompts=prompts) for node in nodes}
    acts = [node.get_value(cache, prompts=prompt_dataset.prompts) for node in nodes]
    if unwrap_list:
        acts = acts[0]
    if return_logits:
        res = acts + [logits]
    else:
        res = acts
    if offload_to_cpu:
        res = [x.cpu() for x in res]
    if clear_cache:
        torch.cuda.empty_cache()
    return res


def get_alive_neurons(encoder, llm, n_activations=1e6, threshold=1e-6):
    dataset = load_dataset(encoder.cfg['dataset_name'], split='train')
    if 'TinyStories' in str(dataset) or 'pile' in str(dataset):
        dataset = dataset['train']
    dataset = dataset.shuffle()

    token_dataset = TextDataset(dataset, llm.to_tokens, encoder.cfg['extraction_batch_size'], drop_last_batch=False,
                                seq_len=encoder.cfg['seq_len'])

    text_dataset_loader = iter(DataLoader(token_dataset, batch_size=None, shuffle=False, num_workers=0))

    encoder.cfg['buffer_size'] = 1e6
    buffer = Buffer(
        llm.cuda(),
        text_dataset_loader,
        **encoder.cfg
    )

    n_batches = n_activations // encoder.cfg['batch_size'] - 1
    n_active_per_neuron = torch.zeros(encoder.cfg['d_hidden']).cuda()
    n_total = 0
    for i, batch in tqdm(enumerate(buffer)):
        batch = batch.cuda()
        feature_acts = encoder.encoder(batch) # [batch, d_hidden]
        n_active_per_neuron += (feature_acts > 1e-7).sum(dim=0)
        n_total += feature_acts.shape[0]
        if i >= n_batches:
            break
    frequencies = n_active_per_neuron / n_total
    # get indices of neurons that are more active than threshold
    indices = torch.where(frequencies > threshold)
    return indices[0]


def l0_per_position(encoder, llm, ioi_loader):
    n_features_active = torch.zeros(next(iter(ioi_loader)).shape[1]).cuda()
    n_total = 0
    llm.cuda()
    encoder.cuda()
    i = 0

    names_filter = lambda name_: encoder.cfg['actv_name'] in name_

    dead_neurons = torch.zeros(encoder.cfg['d_hidden']).cuda()

    for batch in ioi_loader:
        # prepend bos
        batch = batch[:10]
        _, cache = llm.run_with_cache(batch.cuda(), names_filter=names_filter)
        if type(encoder.cfg['head']) == int:
            z = cache[encoder.cfg['actv_name']][:, :, encoder.cfg['head'], :]
        elif encoder.cfg['head'] == 'concat':
            z = cache[encoder.cfg['actv_name']]
            z = z.view(z.shape[0], z.shape[1], -1)
        else:
            z = cache[encoder.cfg['actv_name']]
        feature_acts = encoder.encoder(z)  # [batch, seq, d_hidden]
        dead_neurons += feature_acts.sum(dim=[0, 1])
        feature_acts = feature_acts > 1e-7
        n_features_active += feature_acts.sum(dim=[0, 2])
        n_total += feature_acts.shape[0]
        i += 1
        if i > 100:
            break
    n_features_active = n_features_active / n_total
    return n_features_active


def get_ioi_neurons(encoder, llm, ioi_loader, node, max_batches=1e5, threshold=1e-6):
    n_active_per_neuron = torch.zeros(encoder.cfg['d_hidden']).cuda()
    n_total = 0
    for i, batch in tqdm(enumerate(ioi_loader)):
        actv = run_with_cache(llm, batch, node)
        feature_acts = encoder.encoder(actv) # [batch, d_hidden]
        n_active_per_neuron += (feature_acts > 1e-7).sum(dim=0)
        n_total += feature_acts.shape[0]
        if i >= max_batches:
            break
    frequencies = n_active_per_neuron / n_total
    # get indices of neurons that are more active than threshold
    indices = torch.where(frequencies > threshold)
    return indices[0]


def get_genders(names):
    with open('../data/genders_train.txt', 'r') as f:
        genders_train = f.read().splitlines()
    with open('../data/genders_test.txt', 'r') as f:
        genders_test = f.read().splitlines()
    genders_train.extend(genders_test)
    name_to_gender = {}
    for row in genders_train:
        splits = row.split("'")
        name = splits[1]
        gender = splits[3]
        name_to_gender[name] = gender
    name_to_gender
    is_male = []
    for name in names:
        if name_to_gender[name] == 'M':
            is_male.append(True)
        else:
            is_male.append(False)
    return is_male


class LabeledTensor(torch.Tensor):
    """
    A wrapper for PyTorch tensors that allows labeling and indexing tensor dimensions
    using user-defined names and categorical labels. This class facilitates intuitive
    interaction with tensors in applications where dimensions represent specific
    categories or entities, improving code readability and usability.

    Attributes:
        tensor (torch.Tensor): The underlying tensor.
        maps (dict): A dictionary containing mappings for each labeled dimension.

    Example usage:
        # Initialize a LabeledTensor with custom labels
        tensor = LabeledTensor(tensor_shape=(2, 3, 4), num_neurons=100, device='cuda',
                               measures=['count', 'sum'],
                               nodes=['node1', 'node2'],
                               patterns=['ABB', 'BAB'],
                               roles=['s', 'io'],
                               names=['name1', 'name2', 'name3'])
                               
        # Access and modify the tensor using labeled indices
        tensor['count', 'node1', 'ABB', 's', 'name1'] = 5
        print(tensor['count', 'node1', 'ABB', 's', 'name1'])
    """
    def __new__(cls, data, device='cpu', **labels):
        if type(data) is torch.Size or type(data) is tuple:
            data = torch.zeros(data, dtype=torch.float32, device=device)
        if type(data) is torch.Tensor or type(data) is LabeledTensor:
            return torch.Tensor._make_subclass(cls, data)
        else:
            raise ValueError('Invalid data type. Must be a torch.Tensor or torch.Size. Data type:', type(data))
        

    def __init__(self, tensor_shape, device='cuda', **labels):
        self.maps = {label: {name: i for i, name in enumerate(names)} for label, names in labels.items()}

    def __translate_key(self, key):
        """
        Translates a key from labels to indices, handling mixed types (slices, strings, integers).
        """
        if isinstance(key, tuple):
            mapped_key = list(key)  # Copy to allow modification
            for i, k in enumerate(key):
                if isinstance(k, (slice, int)):
                    continue  # Leave slices and integers as they are
                if isinstance(k, LabeledTensor):
                    if k.numel() == 1:
                        mapped_key[i] = k.item()
                    else:
                        mapped_key[i] = tuple(k.cpu().numpy().tolist())
                for map in self.maps.values():
                    if k in map:
                        mapped_key[i] = map[k]
                        break
            # check if any elements are still other objects
            for k in mapped_key:
                if not isinstance(k, int) and not isinstance(k, slice) and not isinstance(k, tuple) and not isinstance(k, list) and not isinstance(k, torch.Tensor):
                    raise ValueError('Invalid key. Must be a slice, integer, or a label from the tensor maps. Key:', k)
            return tuple(mapped_key)
        else:  # Handle single non-tuple keys
            for map in self.maps.values():
                if key in map:
                    return map[key]
            return key

    def __getitem__(self, key):
        if hasattr(self, 'maps'):
            key = self.__translate_key(key)
            t = super().__getitem__(key)
            t.maps = self.maps
        else:
            t = super().__getitem__(key)
        return t

    def __setitem__(self, key, value):
        if hasattr(self, 'maps'):
            key = self.__translate_key(key)
        super().__setitem__(key, value)

    def sum(self, dim=None, keepdim=False, **kwargs):
        """
        Overridden sum() method that updates the maps attribute for reduced dimensions.
        
        Args:
            dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
            keepdim (bool): Whether the output tensor has dim retained or not.
            **kwargs: Other keyword arguments for the original torch.Tensor.sum() method.
        Returns:
            LabeledTensor: A new LabeledTensor with reduced dimensions.
        """
        # Perform the sum operation on the underlying tensor.
        result_tensor = super().sum(dim=dim, keepdim=keepdim, **kwargs)
        
        # If dim is None, all dimensions are reduced, and the result should have no labels.
        if dim is None and not keepdim:
            new_maps = {}
        else:
            new_maps = self.maps

        # Create a new LabeledTensor for the result with the adjusted maps.
        new_instance = self.__class__(result_tensor)
        new_instance.maps = new_maps  # it's faster (but ugly) to just assign the existing object
        return new_instance


    def __repr__(self):
        return 'LabeledTensor containing:\n' + super().__repr__()

    def save(self, filepath):
        """
        Saves the LabeledTensor to a file, including both the tensor data and the labels.

        Parameters:
        - filepath (str): The path to the file where the tensor and labels will be saved.
        """
        data_to_save = {
            'tensor': self.data,
            'maps': self.maps
        }
        torch.save(data_to_save, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Loads a LabeledTensor from a file.

        Parameters:
        - filepath (str): The path to the file from which to load the tensor and labels.

        Returns:
        - LabeledTensor: An instance of LabeledTensor with the loaded tensor data and labels.
        """
        data_loaded = torch.load(filepath, map_location='cpu')
        tensor = data_loaded['tensor']
        maps = data_loaded['maps']
        
        # Reconstruct the labels from the maps to pass to the constructor
        labels = {dim: list(map.keys()) for dim, map in maps.items()}
        instance = cls(tensor, **labels, device='cpu')
        instance.maps = maps  # Directly assign the loaded maps to ensure indices match
        return instance
    

# All feature function:

# On <end> and <s2> token:
    
def s_name_score(actv_counts, test_actv_counts=None, k=30):
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', :, 's', :].sum(dim=0)  # -> (names,)
    total_activations = actv_counts['total', :, 's', :].sum(dim=0)  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', :, 's', selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', :, 's', selected_names].sum()  # -> ()
        n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', :, 's', best_names_idx].sum()
        total_activations = test_actv_counts['total', :, 's', best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature
        

def contains_name_score(actv_counts, test_actv_counts=None, k=30):
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', :, :, :].sum(dim=(0, 1))  # -> (names,)
    total_activations = actv_counts['total', :, :, :].sum(dim=(0, 1))  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', :, :, selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', :, :, selected_names].sum()  # -> ()

        # correct for double counting
        corr = len(selected_names) / len(names)
        n_active = actv_counts['count', :, :, :].sum() / (2 - corr) + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', :, :, best_names_idx].sum()
        total_activations = test_actv_counts['total', :, :, best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / (2 - best_names_idx.numel() / len(names)) + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


def io_name_score(actv_counts, test_actv_counts=None, k=30):
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', :, 'io', :].sum(dim=0)  # -> (names,)
    total_activations = actv_counts['total', :, 'io', :].sum(dim=0)  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero
    # Why divide by 2?
    # because every correct detection of a name-io feature has a count in the s feature of the other name.
    # so in actv_counts, we count every occurrence of a feature twice and we need to correct for that

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', :, 'io', selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', :, 'io', selected_names].sum()  # -> ()
        n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', :, 'io', best_names_idx].sum()
        total_activations = test_actv_counts['total', :, 'io', best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature

def first_name_score(actv_counts, test_actv_counts=None, k=30):
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', 'ABB', 'io', :] + actv_counts['count', 'BAB', 's', :]  # -> (names,)
    total_activations = actv_counts['total', 'ABB', 'io', :] + actv_counts['total', 'BAB', 's', :]  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', 'ABB', 'io', selected_names].sum() + actv_counts['count', 'BAB', 's', selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', 'ABB', 'io', selected_names].sum() + actv_counts['total', 'BAB', 's', selected_names].sum()  # -> ()
        n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', 'ABB', 'io', best_names_idx].sum() + test_actv_counts['count', 'BAB', 's', best_names_idx].sum()
        total_activations = test_actv_counts['total', 'ABB', 'io', best_names_idx].sum() + test_actv_counts['total', 'BAB', 's', best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


def second_name_score(actv_counts, test_actv_counts=None, k=30):
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', 'ABB', 's', :] + actv_counts['count', 'BAB', 'io', :]  # -> (names,)
    total_activations = actv_counts['total', 'ABB', 's', :] + actv_counts['total', 'BAB', 'io', :]  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', 'ABB', 's', selected_names].sum() + actv_counts['count', 'BAB', 'io', selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', 'ABB', 's', selected_names].sum() + actv_counts['total', 'BAB', 'io', selected_names].sum()  # -> ()
        n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', 'ABB', 's', best_names_idx].sum() + test_actv_counts['count', 'BAB', 'io', best_names_idx].sum()
        total_activations = test_actv_counts['total', 'ABB', 's', best_names_idx].sum() + test_actv_counts['total', 'BAB', 'io', best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


def name_x_pos(actv_counts, pattern, role, test_actv_counts=None, k=30):
    """
    e.g. for "name yxz is subject and at the first position" use pattern='BAB', role='s'
    or for "name XYZ is indirect object and at second position" use pattern='BAB', role='io'
    """
    # shape of actv_counts: (measures, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[1]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[2]
    assert len(actv_counts.maps['names']) == actv_counts.shape[3]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', pattern, role, :]  # -> (names,)
    total_activations = actv_counts['total', pattern, role, :]  # -> (names,)
    n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = actv_counts['count', pattern, role, selected_names].sum()  # -> (); cumulative number of activations across all topk names
        total_activations = actv_counts['total', pattern, role, selected_names].sum()  # -> ()
        n_active = actv_counts['count', :, :, :].sum() / 2 + 1e-8 # -> ()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = test_actv_counts['count', pattern, role, best_names_idx].sum()
        total_activations = test_actv_counts['total', pattern, role, best_names_idx].sum()
        n_active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


# On IO and S1 token

def gender_score(actv_counts, io_node, s1_node, gender, is_male, test_actv_counts=None):
    if gender=='F':
        is_male = [not val for val in is_male]
    gender_names_idx = torch.tensor([i for i, val in enumerate(is_male) if val])

    active_at_pattern = actv_counts['count', io_node, :, 'io', gender_names_idx].sum() + actv_counts['count', s1_node, :, 's', gender_names_idx].sum() 
    total_at_pattern = actv_counts['total', io_node, :, 'io', gender_names_idx].sum() + actv_counts['total', s1_node, :, 's', gender_names_idx].sum()
    active = actv_counts['count', io_node, :, 'io'].sum() + actv_counts['count', s1_node, :, 's'].sum() + 1e-8

    recall = active_at_pattern / total_at_pattern
    precision = active_at_pattern / active
    f_score = get_f_score(precision, recall)

    score = {
        'feature_type': f'current_pos_is_gender_{gender}',
        'topk': 0,
        'names': [],
        'recall': recall.item(),
        'precision': precision.item(),
        'f_score': f_score.item()
    }

    if test_actv_counts is not None:
        active_at_pattern = test_actv_counts['count', io_node, :, 'io', gender_names_idx].sum() + test_actv_counts['count', s1_node, :, 's', gender_names_idx].sum() 
        total_at_pattern = test_actv_counts['total', io_node, :, 'io', gender_names_idx].sum() + test_actv_counts['total', s1_node, :, 's', gender_names_idx].sum()
        active = test_actv_counts['count', io_node, :, 'io'].sum() + test_actv_counts['count', s1_node, :, 's'].sum() + 1e-8

        recall = active_at_pattern / total_at_pattern
        precision = active_at_pattern / active
        f_score = get_f_score(precision, recall)
        score['recall_test'] = recall.item()
        score['precision_test'] = precision.item()
        score['f_score_test'] = f_score.item()
    return score

def name_score(actv_counts, io_node, s1_node, test_actv_counts=None, k=30):
# shape of actv_counts: (measures, nodes, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['ioi_nodes']) == actv_counts.shape[1]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[2]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[3]
    assert len(actv_counts.maps['names']) == actv_counts.shape[4]

    # calculate recall, precision, f_score for each name
    counts_at_name = actv_counts['count', io_node, :, 'io', :].sum(dim=0) + actv_counts['count', s1_node, :, 's', :].sum(dim=0) # -> (names,)
    total_activations = actv_counts['total', io_node, :, 'io', :].sum(dim=0) + actv_counts['total', s1_node, :, 's', :].sum(dim=0)  # -> (names,)
    n_active = (actv_counts['count', io_node, :, :, :].sum() + actv_counts['count', s1_node, :, :, :].sum()) / 2 + 1e-8 # -> (); prevent division by zero

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        counts_at_name = (actv_counts['count', io_node, :, 'io', selected_names].sum() + actv_counts['count', s1_node, :, 's', selected_names].sum()).sum()
        total_activations = (actv_counts['total', io_node, :, 'io', selected_names].sum() + actv_counts['total', s1_node, :, 's', selected_names].sum()).sum()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        counts_at_name = (test_actv_counts['count', io_node, :, 'io', best_names_idx].sum() + test_actv_counts['count', s1_node, :, 's', best_names_idx].sum()).sum()
        total_activations = (test_actv_counts['total', io_node, :, 'io', best_names_idx].sum() + test_actv_counts['total', s1_node, :, 's', best_names_idx].sum()).sum()
        n_active = (test_actv_counts['count', io_node, :, :, :].sum() + test_actv_counts['count', s1_node, :, :, :].sum()) / 2 + 1e-8 # -> (); prevent division by zero

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


def name_x_context_pos_score(actv_counts, io_node, s1_node, position, test_actv_counts=None, k=30):
# shape of actv_counts: (measures, nodes, patterns, roles, names)
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['ioi_nodes']) == actv_counts.shape[1]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[2]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[3]
    assert len(actv_counts.maps['names']) == actv_counts.shape[4]

    # calculate recall, precision, f_score for each name
    if position == 1:
        counts_at_name = actv_counts['count', io_node, 'ABB', 'io', :] + actv_counts['count', s1_node, 'BAB', 's', :] # -> (names,)
        total_activations = actv_counts['total', io_node, 'ABB', 'io', :] + actv_counts['total', s1_node, 'BAB', 's', :]  # -> (names,)
    elif position == 2:
        counts_at_name = actv_counts['count', io_node, 'BAB', 'io', :] + actv_counts['count', s1_node, 'ABB', 's', :]
        total_activations = actv_counts['total', io_node, 'BAB', 'io', :] + actv_counts['total', s1_node, 'ABB', 's', :]
    n_active = actv_counts['count', io_node, 'ABB', 'io'].sum() + actv_counts['count', s1_node, 'BAB', 's'].sum() + actv_counts['count', io_node, 'BAB', 'io'].sum() + actv_counts['count', s1_node, 'ABB', 's'].sum() + 1e-8

    recalls = counts_at_name / total_activations
    precisions = counts_at_name / n_active
    f_scores = get_f_score(precisions, recalls)  # -> (names,)

    # select the best IO name feature according to the f-score
    names = list(actv_counts.maps['names'].keys())
    idx_best = f_scores.argmax()
    best_feature = {
        'topk': 1,
        'names': [names[idx_best]],
        'recall': recalls[idx_best].item(),
        'precision': precisions[idx_best].item(),
        'f_score': f_scores[idx_best].item()
    }
    best_names_idx = idx_best
    
    # if we can get a better f-score by selecting more names, do so
    for k in range(2, k+1):
        selected_names = torch.topk(f_scores, k).indices

        if position == 1:
            counts_at_name = (actv_counts['count', io_node, 'ABB', 'io', selected_names].sum() + actv_counts['count', s1_node, 'BAB', 's', selected_names].sum()).sum()
            total_activations = (actv_counts['total', io_node, 'ABB', 'io', selected_names].sum() + actv_counts['total', s1_node, 'BAB', 's', selected_names].sum()).sum()
        elif position == 2:
            counts_at_name = (actv_counts['count', io_node, 'BAB', 'io', selected_names].sum() + actv_counts['count', s1_node, 'ABB', 's', selected_names].sum()).sum()
            total_activations = (actv_counts['total', io_node, 'BAB', 'io', selected_names].sum() + actv_counts['total', s1_node, 'ABB', 's', selected_names].sum()).sum()

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)  # -> (); f-score for the feature "IO is one of selected_names"
        if f_score > best_feature['f_score']:
            best_feature = {
                'topk': k,
                'names': [names[t.item()] for t in selected_names],
                'recall': recall.item(),
                'precision': precision.item(),
                'f_score': f_score.item()
            }
            best_names_idx = selected_names

    # test
    if test_actv_counts is not None:
        if position == 1:
            counts_at_name = (test_actv_counts['count', io_node, 'ABB', 'io', best_names_idx].sum() + test_actv_counts['count', s1_node, 'BAB', 's', best_names_idx].sum()).sum()
            total_activations = (test_actv_counts['total', io_node, 'ABB', 'io', best_names_idx].sum() + test_actv_counts['total', s1_node, 'BAB', 's', best_names_idx].sum()).sum()
        elif position == 2:
            counts_at_name = (test_actv_counts['count', io_node, 'BAB', 'io', best_names_idx].sum() + test_actv_counts['count', s1_node, 'ABB', 's', best_names_idx].sum()).sum()
            total_activations = (test_actv_counts['total', io_node, 'BAB', 'io', best_names_idx].sum() + test_actv_counts['total', s1_node, 'ABB', 's', best_names_idx].sum()).sum()
        n_active = test_actv_counts['count', io_node, 'ABB', 'io'].sum() + test_actv_counts['count', s1_node, 'BAB', 's'].sum() + test_actv_counts['count', io_node, 'BAB', 'io'].sum() + test_actv_counts['count', s1_node, 'ABB', 's'].sum() + 1e-8

        recall = counts_at_name / total_activations
        precision = counts_at_name / n_active
        f_score = get_f_score(precision, recall)
        best_feature['recall_test'] = recall.item()
        best_feature['precision_test'] = precision.item()
        best_feature['f_score_test'] = f_score.item()
    return best_feature


def io_pos_score(actv_counts, pattern, test_actv_counts=None):
    active_at_pattern = actv_counts['count', pattern, :, :].sum()
    total_at_pattern = actv_counts['total', pattern, :, :].sum()
    active = actv_counts['count', :, :, :].sum() + 1e-8
    recall = active_at_pattern / total_at_pattern
    precision = active_at_pattern / active
    f_score = get_f_score(precision, recall)
    score = {
        'feature_type': f'io_position_{pattern}',
        'topk': 0,
        'names': [],
        'recall': recall.item(),
        'precision': precision.item(),
        'f_score': f_score.item()
    }

    # test
    if test_actv_counts is not None:
        active_at_pattern = test_actv_counts['count', pattern, :, :].sum()
        total_at_pattern = test_actv_counts['total', pattern, :, :].sum()
        active = test_actv_counts['count', :, :, :].sum() + 1e-8
        recall = active_at_pattern / total_at_pattern
        precision = active_at_pattern / active
        f_score = get_f_score(precision, recall)
        score['recall_test'] = recall.item()
        score['precision_test'] = precision.item()
        score['f_score_test'] = f_score.item()
    return score


def gender_x_role_score(actv_counts, role, gender, is_male, test_actv_counts=None):
    if gender=='F':
        is_male = [not val for val in is_male]
    gender_names_idx = torch.tensor([i for i, val in enumerate(is_male) if val])

    active_at_pattern = actv_counts['count', :, role, gender_names_idx].sum()
    total_at_pattern = actv_counts['total', :, role, gender_names_idx].sum()
    active = actv_counts['count', :, :, :].sum() / 2 + 1e-8

    recall = active_at_pattern / total_at_pattern
    precision = active_at_pattern / active
    f_score = get_f_score(precision, recall)

    score = {
        'feature_type': f'{role}_is_gender_{gender}',
        'topk': 0,
        'names': [],
        'recall': recall.item(),
        'precision': precision.item(),
        'f_score': f_score.item()
    }

    if test_actv_counts is not None:
        active_at_pattern = test_actv_counts['count', :, role, gender_names_idx].sum()
        total_at_pattern = test_actv_counts['total', :, role, gender_names_idx].sum()
        active = test_actv_counts['count', :, :, :].sum() / 2 + 1e-8
        recall = active_at_pattern / total_at_pattern
        precision = active_at_pattern / active
        f_score = get_f_score(precision, recall)
        score['recall_test'] = recall.item()
        score['precision_test'] = precision.item()
        score['f_score_test'] = f_score.item()
    return score

def context_position_score(actv_counts, io_node, s1_node, position, test_actv_counts=None):
    assert len(actv_counts.maps['measures']) == actv_counts.shape[0]
    assert len(actv_counts.maps['ioi_nodes']) == actv_counts.shape[1]
    assert len(actv_counts.maps['patterns']) == actv_counts.shape[2]
    assert len(actv_counts.maps['roles']) == actv_counts.shape[3]
    assert len(actv_counts.maps['names']) == actv_counts.shape[4]

    if position == 1:
        active_at_pattern = actv_counts['count', io_node, 'ABB', 'io'].sum() + actv_counts['count', s1_node, 'BAB', 's'].sum()
        total_at_pattern = actv_counts['total', io_node, 'ABB', 'io'].sum() + actv_counts['total', s1_node, 'BAB', 's'].sum()
    elif position == 2:
        active_at_pattern = actv_counts['count', io_node, 'BAB', 'io'].sum() + actv_counts['count', s1_node, 'ABB', 's'].sum()
        total_at_pattern = actv_counts['total', io_node, 'BAB', 'io'].sum() + actv_counts['total', s1_node, 'ABB', 's'].sum()
    active = actv_counts['count', io_node, 'ABB', 'io'].sum() + actv_counts['count', s1_node, 'BAB', 's'].sum() + actv_counts['count', io_node, 'BAB', 'io'].sum() + actv_counts['count', s1_node, 'ABB', 's'].sum()
    active += 1e-8

    recall = active_at_pattern / total_at_pattern
    precision = active_at_pattern / active
    f_score = get_f_score(precision, recall)
    score = {
        'feature_type': f'context_pos_{position}',
        'topk': 0,
        'names': [],
        'recall': recall.item(),
        'precision': precision.item(),
        'f_score': f_score.item()
    }

    # test
    if test_actv_counts is not None:
        if position == 1:
            active_at_pattern = test_actv_counts['count', io_node, 'ABB', 'io'].sum() + test_actv_counts['count', s1_node, 'BAB', 's'].sum()
            total_at_pattern = test_actv_counts['total', io_node, 'ABB', 'io'].sum() + test_actv_counts['total', s1_node, 'BAB', 's'].sum()
        elif position == 2:
            active_at_pattern = test_actv_counts['count', io_node, 'BAB', 'io'].sum() + test_actv_counts['count', s1_node, 'ABB', 's'].sum()
            total_at_pattern = test_actv_counts['total', io_node, 'BAB', 'io'].sum() + test_actv_counts['total', s1_node, 'ABB', 's'].sum()
        active = test_actv_counts['count', io_node, 'ABB', 'io'].sum() + test_actv_counts['count', s1_node, 'BAB', 's'].sum() + test_actv_counts['count', io_node, 'BAB', 'io'].sum() + test_actv_counts['count', s1_node, 'ABB', 's'].sum()
        active += 1e-8

        recall = active_at_pattern / total_at_pattern
        precision = active_at_pattern / active
        f_score = get_f_score(precision, recall)
        score['recall_test'] = recall.item()
        score['precision_test'] = precision.item()
        score['f_score_test'] = f_score.item()
    return score