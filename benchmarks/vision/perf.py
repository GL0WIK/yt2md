"""Isolated vision experiments. Never writes production config, checkpoints or dependencies."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".bench-vision"


def configuration():
    return {"modelsDir": str(ROOT / "models"), "device": "cuda:0",
            "asrModel": "Qwen/Qwen3-ASR-1.7B-hf", "asrAlignerModel": "Qwen/Qwen3-ForcedAligner-0.6B-hf", "visionModel": "Qwen/Qwen3.5-9B",
            "visionQuantization": "nf4", "fps": 1, "groupFrames": 7,
            "audioChunkSeconds": 60, "audioOverlapSeconds": 1, "frameMaxSide": 1280,
            "visionMaxPixels": 393216, "visionMaxNewTokens": 1024}


def prepare(video_id, groups):
    from common import read_json, atomic_json, file_sha256, within_project
    from pipeline import stage_keys
    from runtime import model_locks
    from timestamps import utterances, audio_context
    config = configuration()
    locks = model_locks(config)
    video = within_project(ROOT / "cache" / video_id)
    manifest = read_json(video / "media/frames.json")
    source = read_json(video / "source.json")
    if not manifest or not source:
        raise RuntimeError("Prepared media missing")
    key, _ = stage_keys(source["sha256"], config, locks)
    transcript = []
    for filename in sorted((video / "asr" / key).glob("*.json")):
        record = read_json(filename)
        entries, _ = utterances(record["result"], record["window"])
        transcript.extend(entries)
    for index in groups:
        frames = manifest["frames"][index * 7:(index + 1) * 7]
        if not frames:
            raise ValueError("Group missing")
        target = BENCH / "fixtures" / str(index)
        target.mkdir(parents=True, exist_ok=True)
        paths, hashes = [], []
        for frame in frames:
            original = within_project(video / "media" / frame["path"])
            copied = target / Path(frame["path"]).name
            shutil.copy2(original, copied)
            paths.append(str(copied))
            hashes.append(file_sha256(copied))
        atomic_json(target / "fixture.json", {"videoId": video_id, "group": index,
                    "frames": frames, "paths": paths, "sha256": hashes,
                    "totalFrames": len(manifest["frames"]),
                    "context": audio_context(transcript, frames[0]["startMs"], frames[-1]["endMs"])})
    print(json.dumps({"preparedGroups": groups, "audioWordsPreserved": True}))


class TokenMetrics:
    def __init__(self):
        self.started = time.perf_counter()
        self.first = None
        self.last = None
        self.prompt_seen = False
        self.count = 0
        self.last_id = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        now = time.perf_counter()
        self.first = self.first or now
        self.last = now
        self.last_id = int(value.reshape(-1)[-1])
        self.count += 1

    def end(self):
        pass


def benchmark(args):
    from common import read_json, atomic_json
    from main import local_environment
    from runtime import LocalRuntime, model_locks, video_inputs, generation_schema, parse_visual_output, gpu_fields
    from json_constraint import tokenizer_data, json_prefix
    config = configuration()
    local_environment(config)
    import torch
    if args.profile in ("fla-conv", "fla-fused", "fla-compile"):
        from fast_kernels import install_convolution_kernels
        install_convolution_kernels()
    if args.profile == "fla-fused":
        from fused_norms import install_norm_kernels
        install_norm_kernels()
    runtime = LocalRuntime(config, model_locks(config))
    try:
        start = time.perf_counter()
        runtime.load("vision")
        if args.profile == "fla-compile-forced":
            # Experimental opt-in only in this disposable process. Transformers
            # does not advertise BnB4 compilation, so never apply this by default.
            quantizer = runtime.model.hf_quantizer
            quantizer.__class__ = type("ExperimentalCompileableQuantizer", (type(quantizer),), {"is_compileable": True})
        load_seconds = time.perf_counter() - start
        fixture = read_json(BENCH / "fixtures" / str(args.group) / "fixture.json")
        schema = generation_schema()
        inputs = video_inputs(runtime.processor, [Path(p) for p in fixture["paths"]], fixture["frames"],
                              fixture["context"], fixture["totalFrames"], config, schema).to(runtime.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        decoder = "none" if args.free else args.decoder
        if decoder in ("xgrammar", "xgrammar-stream"):
            import xgrammar as xgr
            from xgrammar.contrib.hf import LogitsProcessor
            info = xgr.TokenizerInfo.from_huggingface(runtime.processor.tokenizer,
                vocab_size=runtime.model.config.text_config.vocab_size,
                stop_token_ids=[runtime.processor.tokenizer.eos_token_id])
            compiled = xgr.GrammarCompiler(info).compile_json_schema(schema, max_whitespace_cnt=2)
        else:
            data = tokenizer_data(runtime.processor.tokenizer)
        for repeat in range(args.runs):
            torch.manual_seed(42 + repeat)
            if decoder not in ("xgrammar", "xgrammar-stream"):
                enforcer = json_prefix(data, schema, prompt_len)
            prefix_seconds, prefix_calls = 0.0, 0
            def prefix(batch, tokens):
                nonlocal prefix_seconds, prefix_calls
                started = time.perf_counter()
                value = enforcer(batch, tokens)
                prefix_seconds += time.perf_counter() - started
                prefix_calls += 1
                return value
            if decoder == "xgrammar":
                processor = LogitsProcessor(compiled)
                class TimedProcessor:
                    def __call__(self, input_ids, scores):
                        nonlocal prefix_seconds, prefix_calls
                        started = time.perf_counter()
                        value = processor(input_ids, scores)
                        prefix_seconds += time.perf_counter() - started
                        prefix_calls += 1
                        return value
                decoder_options = {"logits_processor": [TimedProcessor()]}
            elif decoder == "xgrammar-stream":
                matcher = xgr.GrammarMatcher(compiled)
                cpu_mask = xgr.allocate_token_bitmask(1, info.vocab_size).pin_memory()
                gpu_mask = torch.empty_like(cpu_mask, device=runtime.model.device)
                class StreamingProcessor:
                    def __init__(self):
                        self.accepted = 0

                    def __call__(self, input_ids, scores):
                        nonlocal prefix_seconds, prefix_calls
                        started = time.perf_counter()
                        if stream.count != self.accepted:
                            assert stream.count == self.accepted + 1
                            assert matcher.accept_token(stream.last_id)
                            self.accepted += 1
                        matcher.fill_next_token_bitmask(cpu_mask)
                        gpu_mask.copy_(cpu_mask, non_blocking=True)
                        xgr.apply_token_bitmask_inplace(scores, gpu_mask)
                        prefix_seconds += time.perf_counter() - started
                        prefix_calls += 1
                        return scores
                decoder_options = {"logits_processor": [StreamingProcessor()]}
            elif decoder == "none":
                decoder_options = {}
            else:
                decoder_options = {"prefix_allowed_tokens_fn": prefix}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stream = TokenMetrics()
            extra = {}
            if args.profile in ("fla-compile", "fla-compile-forced"):
                from transformers import CompileConfig
                extra = {"cache_implementation": "static", "compile_config": CompileConfig(fullgraph=False, mode="reduce-overhead")}
            with torch.inference_mode():
                output = runtime.model.generate(**inputs, max_new_tokens=args.max_tokens,
                    do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1,
                    use_cache=True, streamer=stream, eos_token_id=runtime.processor.tokenizer.eos_token_id,
                    pad_token_id=runtime.processor.tokenizer.pad_token_id,
                    **extra,
                    **decoder_options)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - stream.started
            raw = runtime.processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
            valid, validation_error = False, None
            try:
                observation = parse_visual_output(raw, stream.count, args.max_tokens)
                valid = True
            except ValueError as error:
                observation, validation_error = None, str(error)
            receipt = {"profile": args.profile, "decoder": decoder, "freeGeneration": args.free, "repeat": repeat,
                "group": args.group, "fixtureHashes": fixture["sha256"], "inputTokens": prompt_len,
                "outputTokens": stream.count, "loadSeconds": round(load_seconds, 3),
                "firstTokenSeconds": round(stream.first - stream.started, 3),
                "elapsedSeconds": round(elapsed, 3),
                "decodeTokensPerSecond": round((stream.count - 1) / (stream.last - stream.first), 2),
                "prefixSeconds": round(prefix_seconds, 3), "prefixCalls": prefix_calls,
                "validJson": valid, "validationError": validation_error,
                "compiledForwardCreated": getattr(runtime.model, "_compiled_call", None) is not None,
                "observation": observation, "raw": raw, **gpu_fields(torch, config["device"])}
            label = "json" if decoder == "lmformat" else ("free" if decoder == "none" else decoder)
            target = BENCH / "results" / f"{args.profile}-{label}-g{args.group}-r{repeat}.json"
            atomic_json(target, receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k not in ("raw", "observation", "fixtureHashes")} ), flush=True)
    finally:
        runtime.unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare")
    parser.add_argument("--profile", choices=["reference", "fla", "fla-conv", "fla-fused", "fla-compile", "fla-compile-forced"], default="reference")
    parser.add_argument("--kernels-check", action="store_true")
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--decoder", choices=["lmformat", "xgrammar", "xgrammar-stream"], default="lmformat")
    args = parser.parse_args()
    BENCH.mkdir(exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(BENCH / "triton-cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(BENCH / "inductor-cache")
    sys.path.insert(0, str(ROOT / "worker"))
    if args.prepare:
        prepare(args.prepare, [0, 2])
    else:
        if args.profile != "reference" or args.kernels_check:
            sys.path.insert(0, str(BENCH / "deps"))
            os.environ["CC"] = str(BENCH / "deps/triton/runtime/tcc/tcc.exe")
        if args.kernels_check:
            from fast_kernels import verify_numerical_equivalence
            from common import atomic_json
            results = verify_numerical_equivalence()
            from fused_norms import verify_norms
            results += verify_norms()
            atomic_json(BENCH / "results/kernel-equivalence.json", results)
            print(json.dumps(results))
        else:
            try:
                benchmark(args)
            except Exception as error:
                from common import atomic_json
                atomic_json(BENCH / "results" / f"{args.profile}-failure.json",
                            {"profile": args.profile, "errorType": type(error).__name__, "error": str(error)[:4000]})
                raise


if __name__ == "__main__":
    main()
