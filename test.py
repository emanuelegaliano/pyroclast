import os
import numpy as np
from dotenv import load_dotenv

from pyroclast import FileMapRepository, HabitatCriteria, InvasionCriteria


def main():
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data").strip('/"\'')
    print(f"Loading data from: {data_path}\n")

    repo = FileMapRepository(data_path)

    # --- Invasion map ---
    invasion = repo.get(InvasionCriteria())
    p = invasion.data
    print(f"Invasion map  : shape={p.shape}, dtype={p.dtype}")
    print(f"  range       : [{p.min():.4f}, {p.max():.4f}]")
    print(f"  active cells: {np.count_nonzero(p):,}")

    # --- Habitat maps ---
    print()
    habitats = repo.matching(HabitatCriteria())
    print(f"Habitats found: {[h.code for h in habitats]}")
    for h_map in habitats:
        h = h_map.data
        n_c = int(np.sum(h))
        at_risk = int(np.sum((p > 0) & (h > 0)))
        print(f"  [{h_map.code}]  presence={n_c:,} cells  |  at-risk={at_risk:,} cells")


if __name__ == "__main__":
    main()
