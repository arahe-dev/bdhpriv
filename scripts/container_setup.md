# Dev container recreation (ice: exact commands used for the ARM-A overnight runs)
#
# This file exists so the (deleted) `iclr-arm-a` Docker container can be
# rebuilt from scratch on any CUDA-capable Docker host in ~10 minutes.
# Torch wheel download is the only large transfer (~3.5 GB with CUDA deps).
#
# Requirements: Docker Desktop with a CUDA-capable GPU and `--gpus all`
# support (Windows: WSL2 backend).

# 1. Create and start the container (repo mounted at /workspace/iclr-oc)
docker run -d --gpus all --ipc=host --name iclr-arm-a `
  -v C:\iclr-oc:/workspace/iclr-oc `
  -w /workspace/iclr-oc `
  python:3.13-slim sleep infinity

# 2. Install the exact stack (torch 2.11.0+cu128 matches the G4 runtime)
docker exec iclr-arm-a sh -c 'pip install --upgrade pip'
docker exec iclr-arm-a sh -c 'pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128'
docker exec iclr-arm-a sh -c 'pip install numpy pyarrow'

# 3. gcc is required by torch.compile / inductor (triton C driver build)
docker exec iclr-arm-a sh -c 'apt-get update -qq && apt-get install -y -qq gcc'

# 4. Smoke check
docker exec iclr-arm-a python -c "import torch, numpy, pyarrow; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. Run the local preflight (gates + scaled smoke) inside the container
docker exec iclr-arm-a python results/verify_opt3c_all.py --microbatch 1 --steps 4 --warmups 1

# 6. Run the local validator for the single G4 cell (uses the cell's own gates)
docker exec iclr-arm-a python opt/validate_final_cell.py

# 7. Run a benchmark session (local A/B; optional)
# docker exec iclr-arm-a python opt/bench_matrix.py --compile --full --gb 1 --blocks 1024 `
#   --packed single,mixed,four,heavy --variants opt3c_where_b1024,opt3c_nockpt_b1024

# Notes
# - Local GPU is an RTX 4060 Laptop (8 GB, sm_89): use it for mechanism
#   discovery and relative ranking only; never project its absolute tok/s
#   to the G4 (RTX PRO 6000 Blackwell, sm_120, ~95 GB).
# - Host-side powershell equivalent of the line-continuation above: use
#   backticks; on bash replace with backslashes.
# - If Docker Desktop's engine dies (observed once after a Windows Update
#   servicing window): kill stale Docker processes, relaunch Docker Desktop,
#   `docker start iclr-arm-a`, verify `import torch` inside.
