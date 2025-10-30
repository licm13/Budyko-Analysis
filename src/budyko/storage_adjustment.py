"""Storage-adjusted Budyko evaporation utilities.

This module implements the minimum viable API for incorporating storage
change terms (e.g. GRACE-derived anomalies) into the Budyko water balance.
It keeps the computation lightweight while exposing hook points for custom
storage providers and aggregation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


class StorageAdjustmentError(RuntimeError):
    """Raised when storage-adjusted evaporation cannot be computed."""


class StorageChangeProvider(Protocol):
    """Protocol for objects capable of supplying storage change time series.

    Implementations typically fetch data from satellite products (e.g.
    GRACE), land-surface models, or in-situ observations. The protocol keeps
    the surface simple to encourage experimentation during the RFC phase.
    """

    name: str

    def storage_change(self, timestamps: Sequence[Any]) -> np.ndarray:
        """Return storage change values aligned with ``timestamps``."""


Aggregator = Callable[[Sequence[np.ndarray]], np.ndarray]
"""Callable signature for combining multiple storage providers."""


@dataclass(slots=True)
class StorageAdjustmentResult:
    """Container for the storage-adjusted evaporation computation.

    Attributes
    ----------
    precipitation:
        Precipitation depth per time step (millimetres).
    runoff:
        Runoff depth per time step (millimetres).
    storage_change:
        Aggregate storage change for the same time steps.
    actual_evaporation:
        Budyko actual evaporation adjusted for storage (``P - Q - ΔS``).
    residual:
        Maximum absolute imbalance encountered during validation, used for
        diagnostics.
    metadata:
        Additional contextual information (e.g. providers involved).
    """

    precipitation: np.ndarray
    runoff: np.ndarray
    storage_change: np.ndarray
    actual_evaporation: np.ndarray
    residual: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageAdjustmentEngine:
    """Compute storage-adjusted evaporation with provider and hook support."""

    def __init__(
        self,
        *,
        aggregator: Optional[Aggregator] = None,
        allow_negative_evaporation: bool = False,
    ) -> None:
        """Create a storage adjustment engine.

        Parameters
        ----------
        aggregator:
            Optional custom aggregation function. When ``None`` the engine
            sums provider outputs element-wise, keeping ``NaN`` placeholders
            intact.
        allow_negative_evaporation:
            Whether negative evaporation values are allowed. When ``False``
            (the default) the engine raises :class:`StorageAdjustmentError`
            if the computed series dips below ``-1e-9`` mm.
        """

        self._providers: MutableMapping[str, StorageChangeProvider] = {}
        self._aggregator: Aggregator = aggregator or self._default_aggregator
        self._allow_negative_evaporation = allow_negative_evaporation

    @staticmethod
    def _default_aggregator(outputs: Sequence[np.ndarray]) -> np.ndarray:
        """Default aggregation: element-wise sum ignoring ``NaN`` values."""

        if not outputs:
            raise StorageAdjustmentError("No storage change outputs available for aggregation.")
        stacked = np.vstack([np.asarray(item, dtype=float) for item in outputs])
        return np.nansum(stacked, axis=0)

    def register_provider(self, provider: StorageChangeProvider) -> None:
        """Register a provider available for later lookup.

        Raises
        ------
        StorageAdjustmentError
            If another provider with the same name is already registered.
        """

        if provider.name in self._providers:
            raise StorageAdjustmentError(f"Provider '{provider.name}' already registered.")
        self._providers[provider.name] = provider
        LOGGER.debug("Registered storage provider '%s'", provider.name)

    def unregister_provider(self, name: str) -> None:
        """Remove a provider from the registry if present."""

        self._providers.pop(name, None)

    def compute(
        self,
        precipitation: Sequence[float],
        runoff: Sequence[float],
        *,
        timestamps: Optional[Sequence[Any]] = None,
        storage_change: Optional[Sequence[float]] = None,
        providers: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StorageAdjustmentResult:
        """Compute storage-adjusted actual evaporation.

        Parameters
        ----------
        precipitation:
            Precipitation depths (mm) per time step.
        runoff:
            Runoff depths (mm) per time step, aligned with ``precipitation``.
        timestamps:
            Optional labels used to align provider outputs. Required when
            ``providers`` is supplied.
        storage_change:
            Pre-computed storage change per time step. Takes precedence over
            provider outputs when supplied.
        providers:
            Optional list of provider names to query. Their outputs are
            aggregated via ``aggregator``.
        metadata:
            Free-form metadata to attach to the resulting object.

        Returns
        -------
        StorageAdjustmentResult
            Computation artefacts and diagnostics.

        Raises
        ------
        StorageAdjustmentError
            If inputs are misaligned or providers cannot satisfy the request.
        """

        precip_arr = np.asarray(precipitation, dtype=float)
        runoff_arr = np.asarray(runoff, dtype=float)

        if precip_arr.shape != runoff_arr.shape:
            raise StorageAdjustmentError("Precipitation and runoff must share the same shape.")

        if storage_change is not None and providers:
            LOGGER.warning(
                "Both direct storage change and providers specified; providers will be ignored.")

        if storage_change is not None:
            storage_arr = np.asarray(storage_change, dtype=float)
        else:
            storage_arr = self._collect_from_providers(providers or (), timestamps)

        if precip_arr.shape != storage_arr.shape:
            raise StorageAdjustmentError("Storage change series must align with precipitation.")

        actual_evap = precip_arr - runoff_arr - storage_arr
        min_evap = float(np.nanmin(actual_evap))
        if not self._allow_negative_evaporation and min_evap < -1e-9:
            raise StorageAdjustmentError(
                "Computed evaporation contains negative values; review storage inputs or set "
                "allow_negative_evaporation=True."
            )

        mass_balance = precip_arr - runoff_arr - storage_arr - actual_evap
        residual = float(np.nanmax(np.abs(mass_balance))) if mass_balance.size else 0.0

        metadata_out: Dict[str, Any] = {"providers": list(providers or [])}
        if metadata:
            metadata_out.update(dict(metadata))

        return StorageAdjustmentResult(
            precipitation=precip_arr,
            runoff=runoff_arr,
            storage_change=storage_arr,
            actual_evaporation=actual_evap,
            residual=residual,
            metadata=metadata_out,
        )

    def _collect_from_providers(
        self, provider_names: Iterable[str], timestamps: Optional[Sequence[Any]]
    ) -> np.ndarray:
        """Collect and aggregate storage change outputs from providers."""

        outputs: List[np.ndarray] = []
        for name in provider_names:
            provider = self._providers.get(name)
            if provider is None:
                raise StorageAdjustmentError(f"Provider '{name}' is not registered.")
            if timestamps is None:
                raise StorageAdjustmentError("timestamps must be supplied when using providers.")
            result = np.asarray(provider.storage_change(timestamps), dtype=float)
            outputs.append(result)
            LOGGER.debug("Provider '%s' returned storage change series of shape %s", name, result.shape)

        if not outputs:
            raise StorageAdjustmentError("No storage change sources available.")

        length = {output.shape for output in outputs}
        if len(length) != 1:
            raise StorageAdjustmentError("Provider outputs must share the same shape for aggregation.")

        return self._aggregator(outputs)


__all__ = [
    "StorageAdjustmentEngine",
    "StorageAdjustmentError",
    "StorageAdjustmentResult",
    "StorageChangeProvider",
]
