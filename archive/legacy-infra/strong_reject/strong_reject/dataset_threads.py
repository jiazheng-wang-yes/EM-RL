"""Thread-based helpers for dataset row transforms.
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor


def resolve_num_workers(num_workers: int | None, default_max_workers: int = 8) -> int:
    if num_workers is None:
        return max(1, min(default_max_workers, multiprocessing.cpu_count()))

    return max(1, int(num_workers))


def map_rows(rows: list[dict], function, num_workers: int | None = None, default_max_workers: int = 8):
    workers = resolve_num_workers(num_workers, default_max_workers=default_max_workers)
    if workers == 1:
        return [function(row) for row in rows]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, rows))


def dataset_map_rows(dataset, function, num_workers: int | None = None, default_max_workers: int = 8):
    rows = [dataset[i] for i in range(len(dataset))]
    outputs = map_rows(
        rows,
        function,
        num_workers=num_workers,
        default_max_workers=default_max_workers,
    )
    if not outputs:
        return dataset

    mapped_dataset = dataset
    for key in outputs[0]:
        if key in mapped_dataset.column_names:
            mapped_dataset = mapped_dataset.remove_columns(key)
        mapped_dataset = mapped_dataset.add_column(key, [output.get(key) for output in outputs])

    return mapped_dataset
