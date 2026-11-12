"""Pure CPU matcher cost, explicitly excluding CUDA waits and GPU masking."""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".bench-vision/deps"))
sys.path.insert(0, str(ROOT / "worker"))
from common import read_json, atomic_json
from runtime import generation_schema
from transformers import AutoProcessor, AutoConfig
import xgrammar as xgr

lock = read_json(ROOT / "models/models.lock.json")
folder = ROOT / lock["models"]["vision"]["path"]
processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
config = AutoConfig.from_pretrained(folder, local_files_only=True)
info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=config.text_config.vocab_size,
                                        stop_token_ids=[processor.tokenizer.eos_token_id])
compiled = xgr.GrammarCompiler(info).compile_json_schema(generation_schema(), max_whitespace_cnt=2)
raw = json.dumps(read_json(ROOT / ".smoke-tmp/vision/receipt.json")["result"]["observation"], ensure_ascii=False)
ids = processor.tokenizer.encode(raw, add_special_tokens=False)
mask = xgr.allocate_token_bitmask(1, config.text_config.vocab_size)
samples = []
for repeat in range(5):
    matcher = xgr.GrammarMatcher(compiled)
    elapsed = 0
    for token in ids:
        start = time.perf_counter()
        matcher.fill_next_token_bitmask(mask)
        assert matcher.accept_token(token)
        elapsed += time.perf_counter() - start
    samples.append(elapsed)
result = {"tokens": len(ids), "seconds": samples,
          "millisecondsPerToken": [round(s / len(ids) * 1000, 3) for s in samples]}
atomic_json(ROOT / ".bench-vision/results/xgrammar-prefix-cpu.json", result)
print(json.dumps(result))
