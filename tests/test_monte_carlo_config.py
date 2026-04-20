"""Tests for MonteCarloConfig Value Object."""

import pytest

from pyroclast.domain.models import MonteCarloConfig


def _make(**kwargs) -> MonteCarloConfig:
    defaults = {"n_runs": 10_000, "threshold": 0.5, "seed": 42}
    defaults.update(kwargs)
    return MonteCarloConfig(**defaults)


class TestConstruction:
    def test_valid_defaults(self):
        cfg = _make()
        assert cfg.n_runs == 10_000
        assert cfg.threshold == 0.5
        assert cfg.seed == 42

    def test_threshold_zero(self):
        cfg = _make(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_one(self):
        cfg = _make(threshold=1.0)
        assert cfg.threshold == 1.0

    def test_seed_zero(self):
        cfg = _make(seed=0)
        assert cfg.seed == 0

    def test_seed_max_uint32(self):
        cfg = _make(seed=2**32 - 1)
        assert cfg.seed == 2**32 - 1

    def test_n_runs_one(self):
        cfg = _make(n_runs=1)
        assert cfg.n_runs == 1


class TestImmutability:
    def test_frozen(self):
        cfg = _make()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_runs = 999  # type: ignore[misc]

    def test_hashable(self):
        cfg = _make()
        assert hash(cfg) is not None
        assert {cfg: "ok"}[cfg] == "ok"

    def test_equality(self):
        a = _make(n_runs=100, threshold=0.3, seed=7)
        b = _make(n_runs=100, threshold=0.3, seed=7)
        assert a == b

    def test_inequality_n_runs(self):
        assert _make(n_runs=100) != _make(n_runs=200)

    def test_inequality_threshold(self):
        assert _make(threshold=0.3) != _make(threshold=0.7)

    def test_inequality_seed(self):
        assert _make(seed=1) != _make(seed=2)


class TestValidation:
    def test_n_runs_zero(self):
        with pytest.raises(ValueError, match="n_runs"):
            _make(n_runs=0)

    def test_n_runs_negative(self):
        with pytest.raises(ValueError, match="n_runs"):
            _make(n_runs=-1)

    def test_threshold_below_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            _make(threshold=-0.001)

    def test_threshold_above_one(self):
        with pytest.raises(ValueError, match="threshold"):
            _make(threshold=1.001)

    def test_seed_negative(self):
        with pytest.raises(ValueError, match="seed"):
            _make(seed=-1)

    def test_seed_too_large(self):
        with pytest.raises(ValueError, match="seed"):
            _make(seed=2**32)
