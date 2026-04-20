"""Tests for the Monte Carlo service layer (run_monte_carlo, run_monte_carlo_batch).

Uses stub implementations of IMonteCarloAdapter to isolate the service from
any compute backend.
"""

import numpy as np
import pytest

from pyroclast.ABCs.monte_carlo import IMonteCarloAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig
from pyroclast.services.monte_carlo import run_monte_carlo, run_monte_carlo_batch


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FixedAdapter(IMonteCarloAdapter):
    """Adapter that always returns a fixed value and records calls."""

    def __init__(self, return_value: float = 0.75) -> None:
        self.call_count = 0
        self.last_habitat: CompactedHabitat | None = None
        self.last_config: MonteCarloConfig | None = None
        self._return_value = return_value

    def run(self, habitat: CompactedHabitat, config: MonteCarloConfig) -> float:
        self.call_count += 1
        self.last_habitat = habitat
        self.last_config = config
        return self._return_value

    def run_batched(self, habitat, config, n_batches, callback=None) -> float:
        return self.run(habitat, config)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=1_000, threshold=0.5, seed=0)


def _habitat(code: str = "9340", n_cells: int = 5) -> CompactedHabitat:
    p = np.full(n_cells, 0.5, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=n_cells, p_vec=p)


def _empty_habitat(code: str = "empty") -> CompactedHabitat:
    return CompactedHabitat(
        habitat_code=code, n_cells=0, p_vec=np.array([], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_returns_adapter_value(self, config):
        adapter = _FixedAdapter(return_value=0.6)
        result = run_monte_carlo(adapter, _habitat(), config)
        assert result == pytest.approx(0.6)

    def test_delegates_habitat_and_config(self, config):
        adapter = _FixedAdapter()
        hab = _habitat(code="4030")
        run_monte_carlo(adapter, hab, config)
        assert adapter.last_habitat is hab
        assert adapter.last_config is config

    def test_empty_habitat_returns_zero_without_calling_adapter(self, config):
        adapter = _FixedAdapter()
        result = run_monte_carlo(adapter, _empty_habitat(), config)
        assert result == 0.0
        assert adapter.call_count == 0

    def test_result_in_unit_interval(self, config):
        for v in (0.0, 0.5, 1.0):
            adapter = _FixedAdapter(return_value=v)
            result = run_monte_carlo(adapter, _habitat(), config)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# run_monte_carlo_batch
# ---------------------------------------------------------------------------

class TestRunMonteCarloatch:
    def test_returns_dict_keyed_by_habitat_code(self, config):
        adapter = _FixedAdapter(return_value=0.3)
        habs = [_habitat("9340"), _habitat("4030"), _habitat("9110")]
        result = run_monte_carlo_batch(adapter, habs, config)
        assert set(result.keys()) == {"9340", "4030", "9110"}

    def test_values_from_adapter(self, config):
        adapter = _FixedAdapter(return_value=0.42)
        habs = [_habitat("9340"), _habitat("4030")]
        result = run_monte_carlo_batch(adapter, habs, config)
        assert result["9340"] == pytest.approx(0.42)
        assert result["4030"] == pytest.approx(0.42)

    def test_empty_list_returns_empty_dict(self, config):
        adapter = _FixedAdapter()
        result = run_monte_carlo_batch(adapter, [], config)
        assert result == {}
        assert adapter.call_count == 0

    def test_empty_habitat_in_batch_returns_zero(self, config):
        adapter = _FixedAdapter(return_value=0.9)
        habs = [_habitat("9340"), _empty_habitat("ghost")]
        result = run_monte_carlo_batch(adapter, habs, config)
        assert result["ghost"] == 0.0
        assert result["9340"] == pytest.approx(0.9)
        assert adapter.call_count == 1  # only the non-empty habitat called adapter

    def test_call_count_matches_non_empty_habitats(self, config):
        adapter = _FixedAdapter()
        habs = [_habitat("A"), _empty_habitat("B"), _habitat("C"), _empty_habitat("D")]
        run_monte_carlo_batch(adapter, habs, config)
        assert adapter.call_count == 2

    def test_preserves_insertion_order(self, config):
        adapter = _FixedAdapter()
        codes = ["Z", "A", "M", "B"]
        habs = [_habitat(c) for c in codes]
        result = run_monte_carlo_batch(adapter, habs, config)
        assert list(result.keys()) == codes
