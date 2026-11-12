"""CPU-only JSON decoder timing. Does not include CUDA synchronization or model time."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "worker"))
from common import read_json, atomic_json
from runtime import generation_schema
from json_constraint import tokenizer_data, json_prefix
from transformers import AutoProcessor
import torch

lock = read_json(ROOT / "models/models.lock.json")
processor = AutoProcessor.from_pretrained(ROOT / lock["models"]["vision"]["path"], local_files_only=True)
data = tokenizer_data(processor.tokenizer)
receipt = read_json(ROOT / ".smoke-tmp/vision/receipt.json")
raw = json.dumps(receipt["result"]["observation"], ensure_ascii=False)
ids = processor.tokenizer.encode(raw, add_special_tokens=False)
samples = []
for run in range(5):
    prefix = json_prefix(data, generation_schema(), 0)
    elapsed = 0
    for index, token in enumerate(ids):
        current = torch.tensor(ids[:index], dtype=torch.long)
        started = time.perf_counter()
        allowed = prefix(0, current)
        elapsed += time.perf_counter() - started
        assert token in allowed
    assert processor.tokenizer.eos_token_id in prefix(0, torch.tensor(ids))
    samples.append(elapsed)
result = {"tokens": len(ids), "runSeconds": samples,
          "millisecondsPerToken": [round(sample / len(ids) * 1000, 3) for sample in samples],
          "excludesCudaSynchronization": True}
atomic_json(ROOT / ".bench-vision/results/prefix-cpu.json", result)
print(json.dumps(result))
