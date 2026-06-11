# Pyroclast 🌋

GPU-accelerated Monte Carlo simulation library written in Python and OpenCL. It estimates the probability of critical vegetation loss from volcanic lava flow invasions, using per-cell invasion probability maps and habitat masks.

## Features

- **High-Performance Monte Carlo**: Accelerates millions of simulations using the GPU.
- **10+ OpenCL Kernels**: Explores different parallelization strategies including 1D grid-stride, 2D block topologies, vectorized loops, commutative tree reductions, and spatial map-centric approaches.
- **Stream Compaction**: Preprocesses sparse habitat maps directly on the GPU to drastically reduce memory bandwidth and computation time.
- **Cross-Hardware Benchmarking**: Includes a full benchmark suite to test and score performance across different GPUs.

## Requirements

- Python 3.10+
- OpenCL 1.2 or later compatible GPU

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/emanuelegaliano/pyroclast
cd pyroclast
pip install -e .
```

## Running the Demo

You can test the different GPU kernels directly from the command line using the built-in demo script:

```bash
# Run with the default commutative kernel
python demo.py

# Run a specific kernel (e.g., the 2D topology kernel)
python demo.py --monte_carlo 2d --synthetic

# Show all available arguments and kernels
python demo.py --help
```

## Benchmarks & Reports

The repository includes extensive benchmarking scripts in the `benchmark/` directory. 
To view the full analysis of the algorithms, performance scaling, and hardware comparisons, check out the Jupyter Notebooks:

- [`report.ipynb`](report.ipynb) — The complete project report with interactive graphs and algorithm documentation.
- [`report_protanopy.ipynb`](report_protanopy.ipynb) — The same project report featuring a colorblind-friendly (protanopia) color palette.

## Testing

To run the unit tests:
```bash
pytest tests/ --verbose
```
