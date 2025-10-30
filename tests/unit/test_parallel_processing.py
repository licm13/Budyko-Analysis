"""Regression tests for :mod:`src.utils.parallel_processing` refactor."""

from __future__ import annotations

from typing import Dict, List

import pytest

pd = pytest.importorskip("pandas")

from src.utils.parallel_processing import ParallelBudykoAnalyzer, ParallelProcessingError


def _legacy_process_catchments(
    catchment_ids: List[str],
    analysis_function,
    data_loader=None,
    **kwargs,
):
    """Simplified snapshot of the previous implementation for regression tests."""

    results: List[Dict[str, float]] = []
    for catchment_id in catchment_ids:
        try:
            if data_loader is not None:
                data = data_loader(catchment_id)
                payload = analysis_function(catchment_id, data=data, **kwargs)
            else:
                payload = analysis_function(catchment_id, **kwargs)
        except Exception:  # pragma: no cover - snapshot of legacy behaviour
            payload = None

        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("catchment_id", catchment_id)
            results.append(payload)

    return pd.DataFrame(results)


def test_process_catchments_matches_legacy_without_loader():
    ids = ["A", "B", "C"]

    def analysis(catchment_id: str, *, multiplier: float) -> Dict[str, float]:
        return {"value": multiplier * (ord(catchment_id) - 64)}

    analyzer = ParallelBudykoAnalyzer(n_processes=1, verbose=False)
    kwargs = {"multiplier": 1.5}

    legacy_df = _legacy_process_catchments(ids, analysis, **kwargs)
    new_df = analyzer.process_catchments(ids, analysis, **kwargs)

    pd.testing.assert_frame_equal(
        new_df.sort_values("catchment_id").reset_index(drop=True),
        legacy_df.sort_values("catchment_id").reset_index(drop=True),
        check_like=True,
    )


def test_process_catchments_matches_legacy_with_loader():
    ids = ["A", "B", "C"]
    data_store = {cid: (idx,) for idx, cid in enumerate(ids)}

    def loader(catchment_id: str):
        return data_store[catchment_id]

    def analysis(catchment_id: str, *, data) -> Dict[str, float]:
        return {"value": float(data[0]), "catchment_id": catchment_id}

    analyzer = ParallelBudykoAnalyzer(n_processes=1, verbose=False)

    legacy_df = _legacy_process_catchments(ids, analysis, data_loader=loader)
    new_df = analyzer.process_catchments(ids, analysis, data_loader=loader)

    pd.testing.assert_frame_equal(
        new_df.sort_values("catchment_id").reset_index(drop=True),
        legacy_df.sort_values("catchment_id").reset_index(drop=True),
        check_like=True,
    )


def test_process_catchments_records_errors():
    ids = ["ok", "fail"]

    def analysis(catchment_id: str) -> Dict[str, float]:
        if catchment_id == "fail":
            raise ValueError("boom")
        return {"value": 1.0}

    analyzer = ParallelBudykoAnalyzer(n_processes=1, verbose=False)
    df = analyzer.process_catchments(ids, analysis)

    assert df.shape[0] == 1
    assert analyzer.last_errors == ["fail: boom"]


def test_process_catchments_raises_when_all_fail():
    ids = ["a", "b"]

    def analysis(_: str):
        raise RuntimeError("nope")

    analyzer = ParallelBudykoAnalyzer(n_processes=1, verbose=False)

    with pytest.raises(ParallelProcessingError):
        analyzer.process_catchments(ids, analysis)


def test_batch_trajectory_analysis_end_to_end():
    data = pd.DataFrame(
        {
            "catchment_id": ["c1", "c2"],
            "IA_1": [0.5, 0.6],
            "IE_1": [0.3, 0.4],
            "IA_2": [0.7, 0.9],
            "IE_2": [0.35, 0.42],
        }
    )

    analyzer = ParallelBudykoAnalyzer(n_processes=1, verbose=False)
    output = analyzer.batch_trajectory_analysis(data, ("IA_1", "IE_1"), ("IA_2", "IE_2"))

    assert set(output.columns) == {
        "catchment_id",
        "intensity",
        "direction_angle",
        "follows_curve",
        "movement_type",
        "delta_IA",
        "delta_IE",
    }
    assert output.shape[0] == 2
