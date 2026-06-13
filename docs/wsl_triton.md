# Running the fused Triton kernel on Windows (WSL2)

The fused W8A16 dequant-GEMM kernel ([`atlasinfer/triton_kernels.py`](../atlasinfer/triton_kernels.py))
needs Triton, which is Linux + GPU only. On Windows that means **WSL2**, which
exposes the host NVIDIA GPU to a real Linux environment. The numbers in the
README's kernel table were produced this way on an RTX 3060 Laptop.

## Prerequisites

- Windows 10 21H2+ / Windows 11 with WSL2 and an Ubuntu distro
  (`wsl --install -d Ubuntu`).
- A recent NVIDIA Windows driver (it provides the CUDA library to WSL — no CUDA
  toolkit install needed inside WSL). Verify with `wsl nvidia-smi`.

## Setup (3 steps)

```bash
# 1. Build tooling + Python headers (Triton JIT-compiles a small C helper).
sudo apt-get update && sudo apt-get install -y build-essential python3-dev python3-venv

# 2. A venv with CUDA torch + Triton (Triton ships as a torch dependency on Linux).
python3 -m venv ~/atlasvenv
~/atlasvenv/bin/pip install --upgrade pip
~/atlasvenv/bin/pip install torch==2.5.1 numpy

# 3. Run the kernel microbenchmark against the repo on your Windows drive.
cd /mnt/c/path/to/AtlasInfer
~/atlasvenv/bin/python bench_triton_kernel.py
```

Expected output: the fused kernel ~1.4–1.6x faster than FP16 and ~7–8x faster
than the dequant-then-matmul path at batch-1 decode.

## No-sudo fallback for the Python headers

If you can't `sudo apt install python3-dev`, fetch the headers without root by
extracting the `.deb` and pointing gcc at them:

```bash
cd ~ && mkdir -p pydev && cd pydev
apt-get download libpython3.12-dev python3.12-dev      # no root needed
for d in *.deb; do dpkg-deb -x "$d" extracted; done
export C_INCLUDE_PATH="$PWD/extracted/usr/include/python3.12:$PWD/extracted/usr/include"
# now run bench_triton_kernel.py in this shell
```

(Match `python3.12` to your distro's Python version.)
