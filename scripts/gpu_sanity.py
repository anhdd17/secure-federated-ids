"""Phase 0 sanity check: torch sees the accelerator and can run a matmul on it."""
import sys
import time

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    device = pick_device()
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    start = time.perf_counter()
    c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"matmul 4096x4096 on {device}: {elapsed * 1000:.1f} ms, checksum {c.sum().item():.3e}")

    # Exit non-zero when run with --require-cuda (on the GPU server) but CUDA is missing.
    if "--require-cuda" in sys.argv and device.type != "cuda":
        print("FAIL: --require-cuda given but CUDA is not available")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
