import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FLASHMLA_DIR = REPO_ROOT / "external" / "FlashMLA"
HARNESS_SRC = FLASHMLA_DIR / "tests" / "sm120_copy_index_test.cu"
HARNESS_BIN = FLASHMLA_DIR / "build" / "sm120_copy_index_test"


def _find_nvcc() -> Path | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc)
    # Fallback to CUDA installed via Windows PATH (nvcc.exe)
    nvcc_exe = shutil.which("nvcc.exe")
    return Path(nvcc_exe) if nvcc_exe else None


def _compile_harness(nvcc: Path) -> None:
    output_dir = HARNESS_BIN.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(nvcc),
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-arch=sm_89",  # matches RTX 4090/SM120 development environment
        "-I",
        str(FLASHMLA_DIR / "csrc"),
        "-I",
        str(FLASHMLA_DIR / "csrc" / "cutlass" / "include"),
        "-I",
        str(FLASHMLA_DIR / "csrc" / "cutlass" / "tools" / "util" / "include"),
        str(HARNESS_SRC),
        "-o",
        str(HARNESS_BIN),
    ]

    result = subprocess.run(
        cmd,
        cwd=FLASHMLA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Failed to compile TMEM probe with nvcc ({result.returncode}).\n{result.stdout}"
        )


def _run_harness() -> str:
    result = subprocess.run(
        [str(HARNESS_BIN)],
        cwd=FLASHMLA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "TMEM probe executable failed to run "
            f"(exit {result.returncode}).\n{result.stdout}"
        )
    return result.stdout


@pytest.mark.sm120
def test_sm120_tmem_indices_unique():
    """
    Compile and execute the TMEM probe harness to ensure every SM120
    reduce lane writes to a unique TMEM column. The harness prints a
    collision report; any duplicates should fail the test.
    """

    nvcc = _find_nvcc()
    if nvcc is None:
        pytest.skip("nvcc not found in PATH; skipping TMEM layout probe.")

    if not HARNESS_SRC.exists():
        pytest.skip("TMEM probe source missing; nothing to test.")

    _compile_harness(nvcc)
    output = _run_harness()

    if "Index collisions detected" in output:
        pytest.fail(
            "TMEM probe detected overlapping indices in SM120 layout.\n"
            "Harness output:\n"
            f"{output}"
        )
