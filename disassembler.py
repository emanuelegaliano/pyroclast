#!/usr/bin/env python3
"""Disassembler script for pyroclast Monte Carlo kernels.

Compiles the Commutative and Global Seeding Monte Carlo kernels using PyOpenCL
and extracts the compiled hardware binary/PTX code to files for comparison.
"""

import sys
from pathlib import Path
import pyopencl as cl


    # 5. Save output
def save_binary(name: str, binary: bytes, output_dir: Path) -> None:
    # Check if binary is text (like NVIDIA PTX) or binary (like SPIR-V or AMD ELF)
    is_text = False
    decoded_text = ""
    try:
        # Try decoding as UTF-8
        decoded_text = binary.decode("utf-8")
        is_text = True
    except UnicodeDecodeError:
        pass

    if is_text:
        out_path = output_dir / f"{name}.ptx"
        out_path.write_text(decoded_text, encoding="utf-8")
        print(f"Saved assembly text to: {out_path}")
    else:
        out_path = output_dir / f"{name}.bin"
        out_path.write_bytes(binary)
        print(f"Saved raw binary to: {out_path}")

def compile_kernel(kernel_path: Path, mwc64x_include: Path, kernels_dir: Path, ctx) -> bytes:
    source = kernel_path.read_text(encoding="utf-8")
    build_options = f"-I {mwc64x_include} -I {kernels_dir} -DWG_SIZE=256"
    print(f"Compiling {kernel_path.name}...")
    try:
        program = cl.Program(ctx, source).build(options=build_options)
        # program.binaries is a list of bytes, one per device in the context
        return program.binaries[0]
    except Exception as e:
        print(f"Error compiling {kernel_path.name}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    # 1. Setup paths
    workspace_dir = Path(__file__).parent.resolve()
    kernels_dir = workspace_dir / "pyroclast" / "kernels" / "monte_carlo"
    mwc64x_include = workspace_dir / "mwc64x-v0" / "mwc64x" / "cl"
    output_dir = workspace_dir / "disassembled_kernels"
    output_dir.mkdir(exist_ok=True)

    # Kernel paths
    standard_kernel_path = kernels_dir / "monte_carlo.cl"
    global_seed_kernel_path = kernels_dir / "monte_carlo_standard_global_seed.cl"

    for path in (standard_kernel_path, global_seed_kernel_path):
        if not path.is_file():
            print(f"Error: Kernel file not found at {path}", file=sys.stderr)
            sys.exit(1)

    # 2. Get OpenCL context (prefer GPU)
    try:
        platforms = cl.get_platforms()
        devices = []
        for p in platforms:
            devices.extend(p.get_devices(cl.device_type.GPU))
        if not devices:
            for p in platforms:
                devices.extend(p.get_devices())
        if not devices:
            print("Error: No OpenCL devices found.", file=sys.stderr)
            sys.exit(1)
        
        device = devices[0]
        ctx = cl.Context([device])
        print(f"Using device: {device.name} ({device.platform.name})")
    except Exception as e:
        print(f"Error initializing OpenCL context: {e}", file=sys.stderr)
        sys.exit(1)

    standard_binary = compile_kernel(standard_kernel_path, mwc64x_include, kernels_dir, ctx)
    global_seed_binary = compile_kernel(global_seed_kernel_path, mwc64x_include, kernels_dir, ctx)


    save_binary("monte_carlo", standard_binary, output_dir)
    save_binary("monte_carlo_standard_global_seed", global_seed_binary, output_dir)

    print("\nDisassembly complete. Files saved in disassembled_kernels/")


if __name__ == "__main__":
    main()
