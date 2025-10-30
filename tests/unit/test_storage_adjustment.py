"""Tests for storage-adjusted evaporation computations."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from typing import Sequence

from src.budyko.storage_adjustment import StorageAdjustmentEngine, StorageAdjustmentError


class _ArrayProvider:
    """Simple provider used in unit tests."""

    def __init__(self, name: str, values: Sequence[float]) -> None:
        self.name = name
        self._values = np.asarray(values, dtype=float)

    def storage_change(self, timestamps):  # type: ignore[override]
        return self._values


@pytest.fixture()
def engine() -> StorageAdjustmentEngine:
    eng = StorageAdjustmentEngine()
    eng.register_provider(_ArrayProvider("grace", [0.1, -0.2, 0.05]))
    eng.register_provider(_ArrayProvider("model", [0.0, 0.1, 0.0]))
    return eng


def test_compute_with_direct_storage(engine: StorageAdjustmentEngine) -> None:
    result = engine.compute([1.0, 1.2, 1.1], [0.3, 0.5, 0.4], storage_change=[0.1, -0.2, 0.0])
    np.testing.assert_allclose(result.actual_evaporation, [0.6, 0.9, 0.7])
    assert result.metadata["providers"] == []


def test_compute_with_providers(engine: StorageAdjustmentEngine) -> None:
    timestamps = ["t1", "t2", "t3"]
    result = engine.compute([1.0, 1.0, 1.0], [0.2, 0.2, 0.2], timestamps=timestamps, providers=["grace", "model"])
    np.testing.assert_allclose(result.storage_change, [0.1, -0.1, 0.05])
    np.testing.assert_allclose(result.actual_evaporation, [0.7, 0.9, 0.75])
    assert set(result.metadata["providers"]) == {"grace", "model"}


def test_compute_raises_on_negative_evaporation(engine: StorageAdjustmentEngine) -> None:
    with pytest.raises(StorageAdjustmentError):
        engine.compute([0.5, 0.4], [0.6, 0.5], storage_change=[0.0, 0.0])


def test_compute_accepts_negative_when_enabled() -> None:
    engine = StorageAdjustmentEngine(allow_negative_evaporation=True)
    result = engine.compute([0.5, 0.4], [0.6, 0.5], storage_change=[0.0, 0.0])
    np.testing.assert_allclose(result.actual_evaporation, [-0.1, -0.1])


def test_missing_provider_raises(engine: StorageAdjustmentEngine) -> None:
    with pytest.raises(StorageAdjustmentError):
        engine.compute([1, 1, 1], [0.1, 0.1, 0.1], timestamps=["a", "b", "c"], providers=["unknown"])


def test_provider_shape_mismatch(engine: StorageAdjustmentEngine) -> None:
    engine.register_provider(_ArrayProvider("short", [0.0, 0.1]))
    with pytest.raises(StorageAdjustmentError):
        engine.compute([1, 1, 1], [0.1, 0.1, 0.1], timestamps=["a", "b", "c"], providers=["grace", "short"])


