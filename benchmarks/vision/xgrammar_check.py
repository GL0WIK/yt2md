"""Check candidate JSON grammar against valid French and invalid contract examples."""
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
start = time.perf_counter()
grammar = xgr.GrammarCompiler(info).compile_json_schema(generation_schema(), max_whitespace_cnt=2)
valid = {"title": "Écran", "visualDescription": "À gauche, l'écran affiche un prix en euros.",
         "screenText": ["679€"], "uncertainties": []}
cases = [(valid, True), ({**valid, "extra": True}, False), ({"title": "Écran"}, False),
         ({**valid, "visualDescription": "a" * 901}, False), ({**valid, "screenText": ["a"] * 21}, False),
         ({**valid, "title": ""}, False)]
for value, expected in cases:
    matcher = xgr.GrammarMatcher(grammar)
    accepted = matcher.accept_string(json.dumps(value, ensure_ascii=False))
    if accepted:
        accepted = matcher.accept_token(processor.tokenizer.eos_token_id)
    assert accepted == expected, (expected, value)
result = {"passedCases": len(cases), "compileAndCheckSeconds": round(time.perf_counter() - start, 3)}
atomic_json(ROOT / ".bench-vision/results/xgrammar-check.json", result)
print(json.dumps(result))
