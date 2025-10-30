"""Utility helpers for executing Budyko analyses in parallel.

The previous implementation relied on ad-hoc worker functions, global logging
configuration and implicit return contracts.  This refactor preserves the
public API surface of :class:`ParallelBudykoAnalyzer` while introducing a more
robust execution model with explicit result objects, richer error reporting and
type annotations suitable for static analysis.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class ParallelProcessingError(RuntimeError):
    """Raised when one or more catchments fail during parallel execution."""


class InvalidCatchmentResultError(ValueError):
    """Raised when a worker returns an invalid payload."""


CatchmentLoader = Callable[[str], Any]


@dataclass(frozen=True)
class _TaskArguments:
    """Container for task execution parameters passed to worker processes."""

    catchment_id: str
    analysis_function: Callable[..., Mapping[str, Any]]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    data_loader: Optional[CatchmentLoader] = None


@dataclass(slots=True)
class _TaskOutcome:
    """Result produced by a worker."""

    catchment_id: str
    payload: Optional[Mapping[str, Any]]
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.payload is not None


def _execute_with_loader(task: _TaskArguments) -> _TaskOutcome:
    """Worker entry-point when a data loader is provided."""

    try:
        assert task.data_loader is not None, "Data loader required for this worker"
        data = task.data_loader(task.catchment_id)
        payload = task.analysis_function(task.catchment_id, data=data, **task.kwargs)
        return _normalize_payload(task.catchment_id, payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        return _TaskOutcome(catchment_id=task.catchment_id, payload=None, error=str(exc))


def _execute_without_loader(task: _TaskArguments) -> _TaskOutcome:
    """Worker entry-point when analysis functions do not require loading."""

    try:
        payload = task.analysis_function(task.catchment_id, **task.kwargs)
        return _normalize_payload(task.catchment_id, payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        return _TaskOutcome(catchment_id=task.catchment_id, payload=None, error=str(exc))


def _normalize_payload(catchment_id: str, payload: Mapping[str, Any]) -> _TaskOutcome:
    """Validate and normalise worker outputs."""

    if not isinstance(payload, Mapping):
        error = InvalidCatchmentResultError(
            "Analysis function must return a mapping-like object containing results."
        )
        return _TaskOutcome(catchment_id=catchment_id, payload=None, error=str(error))

    result: Dict[str, Any] = dict(payload)
    result.setdefault("catchment_id", catchment_id)
    return _TaskOutcome(catchment_id=catchment_id, payload=result)


class ParallelBudykoAnalyzer:
    """Coordinate Budyko computations across multiple processes.

    Parameters
    ----------
    n_processes:
        Number of worker processes to use.  When set to ``None`` the analyser
        will default to ``cpu_count() - 1`` while ensuring at least a single
        process.
    chunk_size:
        The number of catchments batched per chunk submitted to the process
        pool.
    verbose:
        When ``True`` a progress bar is displayed while iterating over results.
    """

    def __init__(self, n_processes: Optional[int] = None, chunk_size: int = 100, verbose: bool = True) -> None:
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)

        if n_processes < 1:
            raise ValueError("n_processes must be at least 1")

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        self.n_processes = n_processes
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._errors: List[_TaskOutcome] = []

    @property
    def last_errors(self) -> List[str]:
        """Return error messages recorded during the most recent execution."""

        return [f"{outcome.catchment_id}: {outcome.error}" for outcome in self._errors]

    def process_catchments(
        self,
        catchment_ids: Sequence[str],
        analysis_function: Callable[..., Mapping[str, Any]],
        data_loader: Optional[CatchmentLoader] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run an analysis function for multiple catchments in parallel."""

        if not catchment_ids:
            LOGGER.info("No catchments supplied for processing; returning empty frame.")
            return pd.DataFrame()

        self._errors = []
        tasks = self._build_tasks(catchment_ids, analysis_function, data_loader, kwargs)
        outcomes = self._execute_tasks(tasks, data_loader)

        successful_payloads = [outcome.payload for outcome in outcomes if outcome.succeeded]
        self._errors = [outcome for outcome in outcomes if not outcome.succeeded]

        if self._errors:
            LOGGER.warning(
                "%d catchments failed to process: %s",
                len(self._errors),
                ", ".join(self.last_errors),
            )

        if not successful_payloads:
            if self._errors:
                raise ParallelProcessingError(
                    "All catchment analyses failed. Inspect `last_errors` for details."
                )
            return pd.DataFrame()

        return pd.DataFrame(successful_payloads)

    def batch_trajectory_analysis(
        self,
        catchments_df: pd.DataFrame,
        period_1_cols: Tuple[str, str],
        period_2_cols: Tuple[str, str],
    ) -> pd.DataFrame:
        """Convenience wrapper performing trajectory analyses in parallel."""

        from ..budyko.trajectory_jaramillo import TrajectoryAnalyzer

        required_columns = {"catchment_id", *period_1_cols, *period_2_cols}
        missing_columns = required_columns - set(catchments_df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

        analyzer = TrajectoryAnalyzer()

        def analyze_one_catchment(catchment_id: str, *, data: pd.DataFrame, period_1_cols: Tuple[str, str], period_2_cols: Tuple[str, str]) -> Mapping[str, Any]:
            row = data.loc[data["catchment_id"] == catchment_id]
            if row.empty:
                raise KeyError(f"Catchment {catchment_id} not found in provided data.")

            selected = row.iloc[0]
            period_1 = {
                "IA": float(selected[period_1_cols[0]]),
                "IE": float(selected[period_1_cols[1]]),
                "name": "Period_1",
            }
            period_2 = {
                "IA": float(selected[period_2_cols[0]]),
                "IE": float(selected[period_2_cols[1]]),
                "name": "Period_2",
            }

            movement = analyzer.calculate_movement(
                catchment_id=catchment_id,
                period_1=period_1,
                period_2=period_2,
            )

            return {
                "intensity": movement.intensity,
                "direction_angle": movement.direction_angle,
                "follows_curve": movement.follows_curve,
                "movement_type": movement.movement_type,
                "delta_IA": movement.delta_ia,
                "delta_IE": movement.delta_ie,
            }

        return self.process_catchments(
            catchment_ids=catchments_df["catchment_id"].tolist(),
            analysis_function=analyze_one_catchment,
            data_loader=None,
            data=catchments_df,
            period_1_cols=period_1_cols,
            period_2_cols=period_2_cols,
        )

    def _build_tasks(
        self,
        catchment_ids: Sequence[str],
        analysis_function: Callable[..., Mapping[str, Any]],
        data_loader: Optional[CatchmentLoader],
        kwargs: Mapping[str, Any],
    ) -> List[_TaskArguments]:
        tasks: List[_TaskArguments] = []
        for catchment_id in catchment_ids:
            tasks.append(
                _TaskArguments(
                    catchment_id=catchment_id,
                    analysis_function=analysis_function,
                    kwargs=dict(kwargs),
                    data_loader=data_loader,
                )
            )
        return tasks

    def _execute_tasks(
        self,
        tasks: Sequence[_TaskArguments],
        data_loader: Optional[CatchmentLoader],
    ) -> List[_TaskOutcome]:
        if self.n_processes == 1 or len(tasks) <= 1:
            LOGGER.debug("Executing tasks sequentially.")
            executor = _execute_with_loader if data_loader else _execute_without_loader
            return [executor(task) for task in tasks]

        LOGGER.debug("Executing tasks across %d processes.", self.n_processes)
        executor = _execute_with_loader if data_loader else _execute_without_loader

        context = mp.get_context("spawn") if mp.get_start_method(allow_none=True) != "spawn" else mp.get_context()
        with context.Pool(processes=self.n_processes) as pool:
            iterable: Iterable[_TaskOutcome]
            if self.verbose:
                iterable = tqdm(
                    pool.imap(executor, tasks, chunksize=self.chunk_size),
                    total=len(tasks),
                    desc="Analyzing catchments",
                )
            else:
                iterable = pool.imap(executor, tasks, chunksize=self.chunk_size)

            return list(iterable)


__all__ = [
    "ParallelBudykoAnalyzer",
    "ParallelProcessingError",
    "InvalidCatchmentResultError",
]
