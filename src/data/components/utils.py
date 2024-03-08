from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Callable, Optional
import json
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    Normalize,
    CenterCrop,
    InterpolationMode,
)


def map_reduce(
    func: Optional[Callable] = None,
    num_workers: int = 8,
    reduce: str = "none",
    **kwargs,
) -> Any:
    """Map reduce decorator.

    Splits lists into `num_workers` and executes the function in parallel on each chunk. The
    results are then reduced according to the `reduce` argument.

    The data to split is automatically inferred as the longest list in the arguments. If multiple
    lists are passed, it splits all those that have the same length as the longest one. All the
    other lists and arguments are passed as-is to the function.

    Args:
        func (Callable, optional): Function to wrap.
        num_workers (int, optional): Number of workers to use to process the data. If -1, use
            the number of items in the data list. Defaults to 2.
        reduce (str, optional): How to reduce the results. Defaults to "none".
    """
    assert num_workers > 1, "num_workers must be > 1"

    if func is None:
        return partial(map_reduce, num_workers=num_workers, reduce=reduce, **kwargs)

    @wraps(func)
    def inner(*inner_args, **inner_kwargs):
        # iterate over the inner_args to find the longest list
        all_values = inner_args + tuple(inner_kwargs.values())
        size = (
            max([len(values) for values in all_values if isinstance(values, list)]) or 1
        )

        # evaluate the number of samples per process
        items_per_worker = size // num_workers
        item_residual = size % num_workers

        # instantiate the pool executor
        pool = ThreadPoolExecutor if num_workers > 1 else ProcessPoolExecutor

        with pool(max_workers=num_workers) as executor:
            futures = []
            results = [None] * num_workers
            start = 0

            for i in range(num_workers):
                end = start + items_per_worker
                sub_args = [
                    (
                        arg[start:end]
                        if isinstance(arg, list) and len(arg) == size
                        else arg
                    )
                    for arg in inner_args
                ]
                sub_kwargs = {
                    key: (
                        value[start:end]
                        if isinstance(value, list) and len(value) == size
                        else value
                    )
                    for key, value in inner_kwargs.items()
                }

                # submit the task to the executor
                futures.append((i, executor.submit(func, *sub_args, **sub_kwargs)))

                start = end

            if item_residual != 0:
                sub_args = [
                    arg[start:] if isinstance(arg, list) and len(arg) == size else arg
                    for arg in inner_args
                ]
                sub_kwargs = {
                    key: (
                        value[start:]
                        if isinstance(value, list) and len(value) == size
                        else value
                    )
                    for key, value in inner_kwargs.items()
                }

                # Execute the residual task immediately
                assert func is not None, "`func` must be not None"
                results.append(func(*sub_args, **sub_kwargs))
            # gather the results
            for i, future in futures:
                results[i] = future.result()

        # remove empty results
        results = [result for result in results if result is not None and result != []]

        # reduce the results
        if reduce == "sum":
            if isinstance(results[0], list):
                return sum(results, [])
            if isinstance(results[0], tuple):
                num_elements = len(results[0])
                results = [
                    sum([result[i] for result in results], [])
                    for i in range(num_elements)
                ]
                return tuple(results)
            return sum(results)
        elif reduce == "mean":
            return sum(results) / len(results)

        return results

    return inner

def transform():
    return Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data
