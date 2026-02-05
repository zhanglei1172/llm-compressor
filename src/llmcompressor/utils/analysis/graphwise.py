from typing import Dict, Iterable, List, Tuple, Union

import torch
from tqdm import tqdm

from ..measure import (
    METHOD_DISPLAY_NAMES,
    METHOD_USE_PERCENTAGE,
    SUPPORTED_METHODS,
    MeasurePrinter,
    MeasureRecorder,
)


def _normalize_methods(method: Union[str, List[str], None]) -> List[str]:
    """Normalize method parameter to a list of methods.

    Args:
        method: A single method string, list of methods, or None for all methods.

    Returns:
        List of method strings.

    Raises:
        ValueError: If an unsupported method is provided.
    """
    if method is None:
        return SUPPORTED_METHODS.copy()
    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    for m in methods:
        if m not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{m}'. Supported methods: {SUPPORTED_METHODS}"
            )
    return methods


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
            assert isinstance(output_tensor, torch.Tensor), (
                "Output of monitoring operation is not a torch.Tensor"
            )
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
    method: Union[str, List[str], None] = "snr",
    steps: int = 8,
    verbose: bool = True,
    fetchs: int = 4096,
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze quantization error at graph level.

    Args:
        model: The model to analyze.
        dataloader: DataLoader providing input batches.
        method: Measurement method(s). Can be:
            - A single method string (e.g., "snr", "cosine", "mse", "sqnr", "kl")
            - A list of methods (e.g., ["snr", "cosine"])
            - None to use all supported methods
        steps: Number of batches to analyze.
        verbose: Whether to print results.
        fetchs: Number of elements to fetch per batch for comparison.

    Returns:
        If method is a single string: Dict[str, float] mapping layer names to values.
        If method is a list or None: Dict[str, Dict[str, float]] mapping layer names
            to dicts of {method: value}.
    """
    # Determine if single method mode (for backward compatibility)
    single_method_mode = isinstance(method, str)
    methods = _normalize_methods(method)

    # find all quantable operations.
    interested_op: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            interested_op.append((name, operation))

    if len(interested_op) == 0:
        print("Oops. you got nothing to analyse.")
        return {}

    # set up all hooks.
    # recorders[name][method] = MeasureRecorder
    recorders: Dict[str, Dict[str, MeasureRecorder]] = {}
    hooks: Dict[str, OutputRecorder] = {}
    caches: Dict[str, List] = {}
    for name, operation in interested_op:
        if hasattr(operation, "quantization_status"):
            recorders[name] = {m: MeasureRecorder(measurement=m) for m in methods}
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
            hook = hooks[name]
            cache = caches[name]
            y_real = cache[idx]
            y_pred = hook.pop()
            # Update all method recorders
            for m in methods:
                recorders[name][m].update(y_real=y_real, y_pred=y_pred)

        if idx >= steps:
            break

    # Collect results and clean up hooks
    if single_method_mode:
        # Backward compatible: return Dict[str, float]
        results: Dict[str, float] = {}
        for name, operation in interested_op:
            results[name] = recorders[name][methods[0]].measure
            hooks[name].clear()
    else:
        # Multi-method mode: return Dict[str, Dict[str, float]]
        results: Dict[str, Dict[str, float]] = {}
        for name, operation in interested_op:
            results[name] = {m: recorders[name][m].measure for m in methods}
            hooks[name].clear()

    if verbose:
        if single_method_mode:
            # Single method: print once
            m = methods[0]
            method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
            MeasurePrinter(
                results,
                order="large_to_small",
                measure=method_str,
                percentage=METHOD_USE_PERCENTAGE.get(m, False),
            ).print()
        else:
            # Multi-method: print for each method
            for m in methods:
                method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
                # Extract single-method results for printing
                single_results = {name: vals[m] for name, vals in results.items()}
                print(f"\n{'=' * 60}")
                print(f"Method: {m.upper()}")
                print(f"{'=' * 60}")
                MeasurePrinter(
                    single_results,
                    order="large_to_small",
                    measure=method_str,
                    percentage=METHOD_USE_PERCENTAGE.get(m, False),
                ).print()

    return results


@torch.no_grad()
def graph_error_analyse(
    model: torch.nn.Module,
    dataloader: Iterable,
    method: Union[str, List[str], None] = "snr",
    steps: int = 8,
    verbose: bool = True,
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze quantization error at graph level.

    Args:
        model: The model to analyze.
        dataloader: DataLoader providing input batches.
        method: Measurement method(s). Can be:
            - A single method string (e.g., "snr", "cosine", "mse", "sqnr", "kl")
            - A list of methods (e.g., ["snr", "cosine"])
            - None to use all supported methods
        steps: Number of batches to analyze.
        verbose: Whether to print results.

    Returns:
        If method is a single string: Dict[str, float] mapping graph names to values.
        If method is a list or None: Dict[str, Dict[str, float]] mapping graph names
            to dicts of {method: value}.
    """
    # Determine if single method mode (for backward compatibility)
    single_method_mode = isinstance(method, str)
    methods = _normalize_methods(method)

    # find all quantable operations.
    quantable_operations: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status") and getattr(
            operation, "quantization_enabled", True
        ):
            quantable_operations.append((name, operation))

    recorders = {m: MeasureRecorder(measurement=m) for m in methods}

    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="Analysing Graph Overall quantization error:",
        total=min(len(dataloader), steps),
    ):
        # manually override quantization state
        for name, operation in quantable_operations:
            operation.quantization_enabled = False
        fp_outputs = model(batch)

        for name, operation in quantable_operations:
            operation.quantization_enabled = True
        qt_outputs = model(batch)

        # Update all method recorders
        for m in methods:
            recorders[m].update(y_pred=qt_outputs, y_real=fp_outputs)

        if idx >= steps:
            break

    # restore quantization states
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

    name = "graph_error"
    # Collect results
    if single_method_mode:
        # Backward compatible: return Dict[str, float]
        results: Dict[str, float] = {}
        results[name] = recorders[methods[0]].measure
    else:
        # Multi-method mode: return Dict[str, Dict[str, float]]
        results: Dict[str, Dict[str, float]] = {}
        results[name] = {m: recorders[m].measure for m in methods}

    if verbose:
        if single_method_mode:
            # Single method: print once
            m = methods[0]
            method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
            MeasurePrinter(
                results,
                order="large_to_small",
                measure=method_str,
                percentage=METHOD_USE_PERCENTAGE.get(m, False),
            ).print()
        else:
            # Multi-method: print for each method
            for m in methods:
                method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
                # Extract single-method results for printing
                single_results = {name: vals[m] for name, vals in results.items()}
                print(f"\n{'=' * 60}")
                print(f"Method: {m.upper()}")
                print(f"{'=' * 60}")
                MeasurePrinter(
                    single_results,
                    order="large_to_small",
                    measure=method_str,
                    percentage=METHOD_USE_PERCENTAGE.get(m, False),
                ).print()

    return results
