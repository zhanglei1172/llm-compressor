from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from ..measure import MeasurePrinter, MeasureRecorder


def generate_indexer(
    num_of_fetches: int, num_of_elements: int, seed: int = 0x20211230
) -> torch.Tensor:
    """Sample with a given seed. This function will generates a indexer based
    on your seed.

    Args:
        num_of_fetches (int): [description]
        num_of_elements (int): [description]
        seed (int, optional): [description]. Defaults to 0x20211230.

    Returns:
        torch.Tensor: [description]
    """

    indexer = []
    for i in range(num_of_fetches):
        indexer.append(seed % num_of_elements)
        seed = (0x343FD * seed + 0x269EC3) % (2 << 31)
    return torch.tensor(indexer, dtype=torch.int32)


def generate_torch_indexer(num_of_fetches: int, num_of_elements: int) -> torch.Tensor:
    return torch.randint(low=0, high=num_of_elements, size=[num_of_fetches])


def batch_random_fetch(
    tensor: torch.Tensor, fetches_per_batch: int = 1024, seed: int = None
) -> torch.Tensor:
    """Fetch some elements from each sample in a batched tensor. if a valid
    seed is given, elements will be sampled based on your seed, otherwise a
    random seed will be generated.

    result: [num_of_batch, fetchs_per_batch]

    Args:
        tensor (torch.Tensor): [description]
        fetchs_per_channel (int, optional): [description]. Defaults to 1024.

    Returns:
        torch.Tensor: [description]
    """
    tensor = tensor.flatten(start_dim=1)
    num_of_elements = tensor.shape[-1]
    assert num_of_elements > 0, "Can not fetch data from empty tensor(0 element)."

    if seed is None:
        indexer = generate_torch_indexer(
            num_of_fetches=fetches_per_batch, num_of_elements=num_of_elements
        )
    else:
        indexer = generate_indexer(
            num_of_fetches=fetches_per_batch, num_of_elements=num_of_elements, seed=seed
        )
    return tensor.index_select(dim=-1, index=indexer.to(tensor.device).long())


class OutputRecorder:
    def __init__(self, operation: torch.nn.Module, fetchs: int = 4096) -> None:
        self.fetched = None
        self.fetchs = fetchs
        self._hooks = []
        self.operation = operation
        self.register_pre_forward_hook()
        self.register_post_forward_hook()

    def register_pre_forward_hook(self) -> List:
        pass

    def register_post_forward_hook(self) -> List:
        def post_forward_hook(module, args, output):
            output_tensor = output
            assert isinstance(
                output_tensor, torch.Tensor
            ), "Output of monitoring operation is not a torch.Tensor"
            self.fetched = batch_random_fetch(
                output_tensor, seed=10086, fetches_per_batch=self.fetchs
            ).to("cpu")

        hook = self.operation.register_forward_hook(post_forward_hook)
        self._hooks.append(hook)

    def pop(self) -> torch.Tensor:
        fetched = self.fetched
        self.fetched = None
        return fetched

    def clear(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

@torch.no_grad()
def graphwise_error_analyse(
    model: torch.nn.Module,
    dataloader: Iterable,
    method: str = "snr",
    steps: int = 8,
    verbose: bool = True,
    fetchs: int = 4096,
) -> Dict[str, tuple]:
    # find all quantable operations.
    interested_op: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            interested_op.append((name, operation))

    if len(interested_op) == 0:
        print("Oops. you got nothing to analyse.")
        return {}

    # set up all hooks.
    recorders, hooks, caches = {}, {}, {}
    for name, operation in interested_op:
        if hasattr(operation, "quantization_status"):
            recorders[name] = MeasureRecorder(measurement=method)
            hooks[name] = OutputRecorder(operation=operation, fetchs=fetchs)
            caches[name] = []

    # dequantize all
    for name, operation in interested_op:
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = False

    # run for each quantable operations:
    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="Analysing Graphwise Quantization Error(Phrase 1):",
        total=(min(len(dataloader), steps)),
    ):
        batch = batch.to(model.device)
        model(batch)

        for name, operation in interested_op:
            hook = hooks[name]
            caches[name].append(hook.pop())

        if idx >= steps:
            break

    # restore all
    for name, operation in interested_op:
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

    # run for each quantable operations:
    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="Analysing Graphwise Quantization Error(Phrase 2):",
        total=(min(len(dataloader), steps)),
    ):
        batch = batch.to(model.device)
        model(batch)

        for name, operation in interested_op:
            recorder = recorders[name]
            hook = hooks[name]
            cache = caches[name]
            recorder.update(y_real=cache[idx], y_pred=hook.pop())

        if idx >= steps:
            break

    results = {}
    for name, operation in interested_op:
        results[name] = recorders[name].measure
        hooks[name].clear()

    if verbose:
        method_str = "MEASUREMENT"
        if method == "snr":
            method_str = "NOISE:SIGNAL POWER RATIO"
        if method == "cosine":
            method_str = "COSINE SIMILARITY"
        if method == "mse":
            method_str = "MSE LOSS(UNSCALED)"
        MeasurePrinter(
            results,
            order="large_to_small",
            measure=method_str,
            percentage=method in {"snr", "cosine"},
        ).print()
    return results
