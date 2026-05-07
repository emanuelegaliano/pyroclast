import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from pyroclast import (
    FileMapRepository,
    HabitatCriteria,
    InvasionCriteria,
    PyOpenCLAdapter,
    PyOpenCLMonteCarloAdapter,
    PyOpenCLMonteCarloPingPongAdapter,
)
from pyroclast.domain.models import CompactedHabitat, MonteCarloConfig
from pyroclast.services import run_preprocessing_batch


def main() -> None:
    load_dotenv()

    # --- CONFIGURAZIONE IPERPARAMETRI (Modificabili qui) ---
    RUNS_MIN = 100_000
    RUNS_MAX = 1_000_000
    INCREMENT = 50_000
    
    N_BATCHES = 10
    THRESHOLD = 0.005
    SEED = 42
    
    SAVE_FIGURE = True
    # -------------------------------------------------------

    data_path = os.getenv("DATA_PATH", "data").strip('"\'')
    cache_dir = Path(os.getenv("CACHE_DIR", str(Path(data_path) / "cache")).strip('"\''))
    cache_dir.mkdir(parents=True, exist_ok=True)

    invasion_map = os.getenv("INVASION_MAP", "").strip('"\'') or None

    compacted = []
    try:
        repo = FileMapRepository(data_path, invasion_map=invasion_map)
        preprocess_adapter = PyOpenCLAdapter()
        compacted = run_preprocessing_batch(
            repo=repo,
            compute=preprocess_adapter,
            criteria=HabitatCriteria(),
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"Errore durante il caricamento dei dati reali: {e}")
        return

    if not compacted:
        print("Errore: Nessun habitat trovato nel percorso specificato.")
        return

    target_habitat = compacted[0]
    print(f"\nTarget Habitat: {target_habitat.habitat_code} ({target_habitat.n_cells} celle)")

    mc_std = PyOpenCLMonteCarloAdapter(profiling=True)
    mc_pp = PyOpenCLMonteCarloPingPongAdapter(profiling=True)

    run_sizes = list(range(RUNS_MIN, RUNS_MAX + 1, INCREMENT))
    times_std = []
    times_pp = []

    print(f"Inizio benchmark ({RUNS_MIN} -> {RUNS_MAX}, step {INCREMENT})...")

    for runs in run_sizes:
        config = MonteCarloConfig(n_runs=runs, threshold=THRESHOLD, seed=SEED)
        actual_batches = N_BATCHES if runs % N_BATCHES == 0 else 1

        print(f"  Simulazione: MC_RUNS = {runs:<10}", end="\r", flush=True)

        # Standard Kernel
        mc_std.reset_profile()
        mc_std.run_batched(target_habitat, config, actual_batches)
        bench_std = mc_std.benchmark()[0]
        times_std.append(bench_std.mean_ms * bench_std.n_runs)

        # Ping-Pong Kernel
        mc_pp.reset_profile()
        mc_pp.run_batched(target_habitat, config, actual_batches)
        bench_pp = mc_pp.benchmark()[0]
        times_pp.append(bench_pp.mean_ms * bench_pp.n_runs)

    print("\n\nBenchmark completato. Generazione del grafico...")

    plt.figure(figsize=(10, 6))
    plt.plot(run_sizes, times_std, marker='o', linestyle='-', label="Standard 1-D Kernel", color='#1f77b4', linewidth=2)
    plt.plot(run_sizes, times_pp, marker='s', linestyle='-', label="Ping-Pong Kernel", color='#ff7f0e', linewidth=2)
    
    plt.title(f"Benchmark Prestazioni Kernel OpenCL\n(Habitat: {target_habitat.habitat_code}, N_c={target_habitat.n_cells})", fontsize=14)
    plt.xlabel("Numero di simulazioni (MC_RUNS)", fontsize=12)
    plt.ylabel("Tempo Totale Kernel (ms)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if SAVE_FIGURE:
        output_png = "mc_kernels_performance.png"
        plt.savefig(output_png, dpi=150)
        print(f"Grafico salvato in: {output_png}")
    else:
        print("Salvataggio immagine disabilitato (SAVE_FIGURE=False).")
        plt.show()

if __name__ == "__main__":
    main()
