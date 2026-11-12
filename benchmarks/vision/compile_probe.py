"""Small CUDA compilation probe, isolated from production model and caches."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".bench-vision"
sys.path.insert(0, str(BENCH / "deps"))
sys.path.insert(0, str(ROOT / "worker"))
os.environ["CC"] = str(BENCH / "deps/triton/runtime/tcc/tcc.exe")
os.environ["TRITON_CACHE_DIR"] = str(BENCH / "triton-cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(BENCH / "inductor-cache")
from common import atomic_json
import torch

def operation(x):
    return torch.nn.functional.silu(x) * x

try:
    x = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(operation, fullgraph=True, mode="reduce-overhead")
    with torch.inference_mode():
        torch.testing.assert_close(compiled(x), operation(x), atol=0.005, rtol=0.01)
    result = {"supported": True}
except Exception as error:
    result = {"supported": False, "error": str(error)[:2000]}
atomic_json(BENCH / "results/compile-probe.json", result)
print(result)
