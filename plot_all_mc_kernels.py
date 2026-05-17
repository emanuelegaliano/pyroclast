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

    # Calcolo differenza relativa: (T2 - T1) / T1 * 100
    # Dove T1 è Standard (1 barriera) e T2 è Ping-Pong (2 barriere)
    relative_diff = [(pp - std) / std * 100 for std, pp in zip(times_std, times_pp)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Tempi Assoluti
    ax1.plot(run_sizes, times_std, marker='o', linestyle='-', label="Standard 1-D Kernel (1 barriera)", color='#1f77b4', linewidth=2)
    ax1.plot(run_sizes, times_pp, marker='s', linestyle='-', label="Ping-Pong Kernel (2 barriere)", color='#ff7f0e', linewidth=2)
    ax1.set_ylabel("Tempo Totale Kernel (ms)", fontsize=12)
    ax1.set_title(f"Benchmark Prestazioni Kernel OpenCL\n(Habitat: {target_habitat.habitat_code}, N_c={target_habitat.n_cells})", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    # Plot 2: Differenza Relativa
    ax2.plot(run_sizes, relative_diff, marker='d', linestyle='--', color='#2ca02c', linewidth=2, label="(T2-T1)/T1 %")
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)  # Linea di riferimento a 0
    ax2.set_xlabel("Numero di simulazioni (MC_RUNS)", fontsize=12)
    ax2.set_ylabel("Delta Relativo (%)", fontsize=12)
    ax2.set_title("Differenza Relativa: (PingPong - Standard) / Standard", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Aggiunta annotazioni per facilitare l'interpretazione
    if relative_diff:
        ax2.text(run_sizes[0], max(relative_diff), " ↑ Ping-Pong più lento", color='red', verticalalignment='bottom', alpha=0.7)
        ax2.text(run_sizes[0], min(relative_diff), " ↓ Ping-Pong più veloce", color='green', verticalalignment='top', alpha=0.7)

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
