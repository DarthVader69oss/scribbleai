"""
ScribbleAI — environment setup / sanity script.

Run this first, every session (especially on Colab, where the runtime resets).
It does three things:
  1. Confirms a GPU is actually attached before you write any training code.
  2. Confirms every required library imports cleanly.
  3. Prints a short environment report you can paste into your first W&B run notes.

Usage (Colab cell or local terminal):
    python setup_env.py
"""

import importlib
import sys


REQUIRED_LIBRARIES = [
    "torch",
    "numpy",
    "scipy",
    "lxml",
    "pandas",
    "jiwer",
    "Levenshtein",       # import name for python-Levenshtein
    "spellchecker",       # import name for pyspellchecker
    "wandb",
    "gradio",
    "yaml",                # import name for pyyaml
    "sklearn",
    "fastapi",
    "uvicorn",
]

# Nice to have; the project runs without them.
OPTIONAL_LIBRARIES = {
    "pyctcdecode": "pip install --no-deps pyctcdecode   (never plain pip install: it downgrades numpy)",
    "kenlm": "only for the optional pyctcdecode word-LM comparison",
}


def check_gpu():
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed. Run: pip install torch")
        return False

    available = torch.cuda.is_available()
    if available:
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[OK] GPU detected: {name} ({mem_gb:.1f} GB)")
    else:
        print(
            "[WARN] No GPU detected. On Colab: Runtime -> Change runtime type -> "
            "Hardware accelerator -> T4 GPU. Training will be extremely slow on CPU."
        )
    return available


def check_libraries():
    missing = []
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[FAIL] {lib} not importable")
            missing.append(lib)

    if missing:
        print(
            "\nMissing libraries detected. Install with:\n"
            f"  pip install {' '.join(m if m != 'Levenshtein' else 'python-Levenshtein' for m in missing)}\n"
            "Note: kenlm/pyctcdecode are optional, see README.md."
        )
    for lib, hint in OPTIONAL_LIBRARIES.items():
        try:
            importlib.import_module(lib)
            print(f"[OK] {lib} (optional)")
        except ImportError:
            print(f"[--] {lib} not installed (optional: {hint})")
    try:
        import numpy, scipy.sparse  # noqa: F401  (a numpy downgrade breaks scipy imports)
    except Exception as e:  # pragma: no cover
        print(f"[FAIL] numpy/scipy mismatch: {e}. Run: pip install -U numpy scipy, then restart the runtime")
        return False
    return len(missing) == 0


def main():
    print("=" * 60)
    print("ScribbleAI environment check")
    print("=" * 60)

    print(f"\nPython version: {sys.version}\n")

    print("-- GPU check --")
    gpu_ok = check_gpu()

    print("\n-- Library check --")
    libs_ok = check_libraries()

    print("\n" + "=" * 60)
    if gpu_ok and libs_ok:
        print("Environment ready. Proceed to Phase 1 (data pipeline).")
    else:
        print("Fix the issues above before starting Phase 1.")
    print("=" * 60)


if __name__ == "__main__":
    main()
