import json
import platform
import subprocess
import torch

report = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": [p.major, p.minor],
        "total_memory_bytes": int(p.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,temperature.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True
        )
        report["nvidia_smi"] = smi.stdout.strip()
    except Exception as e:
        report["nvidia_smi_error"] = repr(e)

print(json.dumps(report, indent=2))
