"""Independent HF Tiny reference run; never changes the standard suite artifacts."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", default="fleurs_zh=945,fleurs_en=647,mmau=64,mmar=64,mmsu=64")
    parser.add_argument("--mode", choices=["chat", "bare_asr"], default="chat")
    parser.add_argument("--wait-for-suite", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    requested = dict((k, int(v)) for k, v in (x.split("=") for x in args.counts.split(",")))
    all_rows = [json.loads(line) for line in args.manifest.open()]
    selected = []
    # Selection depends only on example IDs, never predictions or correctness.
    for name, count in requested.items():
        rows = [r for r in all_rows if r["benchmark"] == name]
        rows.sort(key=lambda r: hashlib.sha256(("hf-parity-v1:" + r["id"]).encode()).hexdigest())
        if len(rows) < count:
            raise ValueError(f"Insufficient examples for {name}")
        selected.extend(rows[:count])
    if args.mode == "bare_asr" and any(not r["benchmark"].startswith("fleurs_") for r in selected):
        raise ValueError("bare_asr mode is only for ASR diagnostics")
    identity = {"model": str(args.model.resolve()), "manifest": str(args.manifest.resolve()),
                "manifest_sha256": digest(args.manifest), "script_sha256": digest(__file__),
                "model_code": {p.name: digest(p) for p in args.model.glob("*.py")},
                "mode": args.mode, "counts": requested, "ids": [r["id"] for r in selected],
                "packages": {p: importlib.metadata.version(p) for p in
                             ["torch", "transformers", "fla-core", "openai-whisper"]},
                "settings": {"dtype": "bfloat16", "attention": "eager", "do_sample": False,
                             "seed": 42, "preserve_source_fp32": ["A_log", "dt_bias", "expert_bias"],
                             "generation_budget_and_ngram": "from manifest",
                             "rotary_buffers": "validated and rebuilt from source rope initializer"}}
    run_path = args.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != identity:
        raise ValueError("Diagnostic identity changed; use a new directory")
    write_json(run_path, identity)
    saved = {}
    journal_path = args.output / "predictions.jsonl"
    if journal_path.exists():
        for line in journal_path.open():
            row = json.loads(line)
            if row["id"] in saved:
                raise ValueError("Duplicate prediction")
            saved[row["id"]] = row
    if not set(saved).issubset(identity["ids"]):
        raise ValueError("Unexpected saved IDs")
    if len(saved) == len(selected):
        print("Already complete", flush=True)
        return
    if args.wait_for_suite:
        started = time.monotonic()
        while True:
            state = json.loads(args.wait_for_suite.read_text())
            if state["state"] == "complete":
                break
            if state["state"] == "failed":
                raise RuntimeError(f"Standard suite failed: {state}")
            if time.monotonic() - started > 21600:
                raise TimeoutError("Standard suite did not finish within six hours")
            write_json(args.output / "progress.json", {"state": "waiting_for_suite", "suite": state})
            print("Waiting for standard suite", flush=True)
            time.sleep(30)
    # The standard suite stops its owned server before writing complete.
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                          "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free_mib < 30000:
        raise RuntimeError(f"GPU is occupied: only {free_mib} MiB free")
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], str(args.model),
                                        local_files_only=True)
    cls._keep_in_fp32_modules_strict = {"A_log", "dt_bias", "expert_bias"}
    print("Loading original Tiny BF16 with HF eager attention", flush=True)
    model = cls.from_pretrained(args.model, config=config, trust_remote_code=True,
                                local_files_only=True, dtype=torch.bfloat16,
                                attn_implementation="eager", device_map={"": 0}).eval()
    model.requires_grad_(False)
    rotary_audit = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if type(module).__name__ == "BailingMoeV3RotaryEmbedding":
                expected, scaling = module.rope_init_fn(module.config, module.inv_freq.device)
                rotary_audit.append({"module": name, "matched_initializer":
                                     bool(torch.equal(module.inv_freq, expected))})
                module.register_buffer("inv_freq", expected, persistent=False)
                module.original_inv_freq = module.inv_freq
                module.attention_scaling = scaling
    write_json(args.output / "loader-audit.json", {"rotary_buffers": rotary_audit,
               "fp32_parameters": {n: str(p.dtype) for n, p in model.named_parameters()
                                   if any(n.endswith(x) for x in ["A_log", "dt_bias", "expert_bias"])}})
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    initial = len(saved)
    with journal_path.open("a") as journal:
        for row in selected:
            if row["id"] in saved:
                continue
            if digest(row["audio"]["path"]) != row["audio"]["sha256"]:
                raise ValueError("Audio checksum mismatch")
            if args.mode == "bare_asr":
                inputs = processor(row["audio"]["path"], text=row["prompt"])
            else:
                inputs = processor([{"role": "user", "content": [
                    {"type": "audio", "audio": row["audio"]["path"]},
                    {"type": "text", "text": row["prompt"]}]}])
            prompt_ids = inputs["input_ids"][0].tolist()
            inputs = inputs.to(model.device)
            inputs["audio_feats"] = inputs["audio_feats"].to(model.dtype)
            tick = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=row["max_tokens"],
                                        no_repeat_ngram_size=row["ngram"], use_cache=True)
            tokens = output[0, len(prompt_ids):].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            result = {"id": row["id"], "benchmark": row["benchmark"],
                      "text": processor.decode(tokens, skip_special_tokens=True).strip(),
                      "finish_reason": "stop" if tokens[-1] in eos else "length",
                      "elapsed_seconds": time.monotonic() - tick, "prompt_tokens": len(prompt_ids),
                      "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                      "output_ids": tokens}
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            saved[row["id"]] = result
            elapsed = time.monotonic() - started
            rate = (len(saved) - initial) / elapsed
            progress = {"state": "complete" if len(saved) == len(selected) else "running",
                        "completed": len(saved), "total": len(selected), "rate": rate,
                        "eta_seconds": (len(selected) - len(saved)) / rate,
                        "counts": dict(Counter(r["benchmark"] for r in saved.values()))}
            write_json(args.output / "progress.json", progress)
            print(json.dumps({**progress, "last_id": row["id"], "last_text": result["text"][:160]},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
