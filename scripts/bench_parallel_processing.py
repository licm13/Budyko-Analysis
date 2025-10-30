"""Baseline benchmark for :mod:`src.utils.parallel_processing`.

Run the script directly to obtain timing information for sequential versus
parallel execution paths.  The benchmark intentionally keeps the dataset
lightweight so that it can run quickly inside CI environments.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import random
from src.utils.parallel_processing import ParallelBudykoAnalyzer


def _build_fake_dataset(size: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    population = list(range(10_000))
    ids = rng.sample(population, k=min(size, len(population)))
    return [f"CATCHMENT_{idx:05d}" for idx in ids]


def _analysis_function(catchment_id: str, *, scale: float) -> Dict[str, float]:
    return {"value": scale * (hash(catchment_id) % 1000) / 1000.0}


def benchmark(size: int, processes: int, scale: float) -> None:
    ids = _build_fake_dataset(size=size)
    analyzer = ParallelBudykoAnalyzer(n_processes=processes, verbose=False)

    start = time.perf_counter()
    analyzer.process_catchments(ids, _analysis_function, scale=scale)
    duration = time.perf_counter() - start
    print(f"Processed {size} catchments with {processes} process(es) in {duration:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ParallelBudykoAnalyzer")
    parser.add_argument("--size", type=int, default=2000, help="Number of synthetic catchments")
    parser.add_argument("--processes", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--scale", type=float, default=1.2, help="Scaling factor for synthetic analysis")
    args = parser.parse_args()

    benchmark(size=args.size, processes=args.processes, scale=args.scale)


if __name__ == "__main__":  # pragma: no cover - manual benchmarking entrypoint
    main()
