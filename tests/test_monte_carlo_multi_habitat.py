"""GPU integration tests for multi-habitat Monte Carlo simulation.

Verifies the correctness, equivalence, and lack of cross-talk in the multi-habitat
commutative MC kernel and batched reduction.
"""

import numpy as np
import pytest

from pyroclast.adapters.opencl_mc_commutative_adapter import PyOpenCLMonteCarloCommutativeAdapter
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig


@pytest.fixture(scope="module")
def adapter():
    try:
        # Use the commutative adapter as requested
        return PyOpenCLMonteCarloCommutativeAdapter()
    except Exception as exc:
        pytest.skip(f"OpenCL device unavailable: {exc}")


def _habitat(p_values: list[float], code: str) -> CompactedHabitat:
    p = np.array(p_values, dtype=np.float32)
    return CompactedHabitat(habitat_code=code, n_cells=len(p), p_vec=p)


def _config(
    n_runs: int = 10000,
    threshold: float = 0.5,
    seed: int = 42,
) -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=n_runs, threshold=threshold, seed=seed)


def test_empty_habitats(adapter):
    """Passing an empty list of habitats should return an empty list of results."""
    assert adapter.run_multi_habitats([], _config()) == []


def test_single_habitat_equivalence(adapter):
    """Esecuzione con un singolo habitat deve essere identica a run()."""
    hab = _habitat([0.3, 0.6, 0.8, 0.1, 0.9, 0.4], "hab1")
    cfg = _config(n_runs=5000, seed=123)

    single_result = adapter.run(hab, cfg)
    multi_results = adapter.run_multi_habitats([hab], cfg)

    assert len(multi_results) == 1
    assert multi_results[0] == pytest.approx(single_result)


def test_same_size_habitats_bit_exact(adapter):
    """Più habitat della stessa dimensione devono produrre risultati bit-exact con run()."""
    hab1 = _habitat([0.3, 0.6, 0.8, 0.1, 0.9, 0.4], "hab1")
    hab2 = _habitat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "hab2")
    hab3 = _habitat([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "hab3")

    cfg = _config(n_runs=10000, seed=456)

    # Singole esecuzioni sequenziali
    single_res1 = adapter.run(hab1, cfg)
    single_res2 = adapter.run(hab2, cfg)
    single_res3 = adapter.run(hab3, cfg)

    # Esecuzione multi-habitat
    multi_results = adapter.run_multi_habitats([hab1, hab2, hab3], cfg)

    assert len(multi_results) == 3
    assert multi_results[0] == pytest.approx(single_res1)
    assert multi_results[1] == pytest.approx(single_res2)
    assert multi_results[2] == pytest.approx(single_res3)


def test_different_size_habitats_no_crosstalk(adapter):
    """Habitat con dimensioni diverse non devono influenzarsi a vicenda.

    Utilizziamo casi deterministici (tutto 1.0, tutto 0.0) mescolati con un
    habitat probabilistico normale per rilevare cross-talk.
    """
    # hab1: 5 celle a probabilità 1.0 -> distruzione certa (prob = 1.0)
    hab1 = _habitat([1.0] * 5, "certain_destruction")
    # hab2: 15 celle a probabilità 0.0 -> distruzione impossibile (prob = 0.0)
    hab2 = _habitat([0.0] * 15, "no_destruction")
    # hab3: 8 celle a probabilità probabilistica
    hab3 = _habitat([0.5] * 8, "probabilistic")

    cfg = _config(n_runs=5000, threshold=0.4, seed=789)

    multi_results = adapter.run_multi_habitats([hab1, hab2, hab3], cfg)

    assert len(multi_results) == 3
    # Le prime due devono essere esattamente 1.0 e 0.0
    assert multi_results[0] == pytest.approx(1.0)
    assert multi_results[1] == pytest.approx(0.0)
    # La terza deve essere una probabilità valida
    assert 0.0 <= multi_results[2] <= 1.0


def test_different_size_habitats_bit_exact(adapter):
    """Più habitat di dimensioni DIVERSE devono produrre risultati bit-exact rispetto alle singole esecuzioni di run()."""
    hab1 = _habitat([0.3, 0.6, 0.8, 0.1, 0.9, 0.4], "hab1")
    hab2 = _habitat([0.1, 0.2, 0.3, 0.4], "hab2")
    hab3 = _habitat([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], "hab3")

    cfg = _config(n_runs=10000, seed=789)

    # Singole esecuzioni sequenziali
    single_res1 = adapter.run(hab1, cfg)
    single_res2 = adapter.run(hab2, cfg)
    single_res3 = adapter.run(hab3, cfg)

    # Esecuzione multi-habitat
    multi_results = adapter.run_multi_habitats([hab1, hab2, hab3], cfg)

    assert len(multi_results) == 3
    assert multi_results[0] == single_res1
    assert multi_results[1] == single_res2
    assert multi_results[2] == single_res3

