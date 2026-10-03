"""Phases 12-14 (local, CPU + the 4 GB GPU): merge LoRA -> 16-bit HF model -> GGUF -> quantized GGUFs.

  1. merge   base fp16 weights + LoRA adapter (HF repo or local dir) on CPU, saved as fp16 safetensors
             (skipped for the untrained base model)
  2. convert llama.cpp's convert_hf_to_gguf.py (separate env: tools/.venv-convert) -> <tag>-f16.gguf
  3. quantize llama-quantize to each --quants type
  4. imatrix (optional) importance matrix from a calibration file of *training* examples (never test), then an
             imatrix-weighted Q4_K_M, to compare against plain Q4_K_M

Everything lives under models/<tag>/ on D: (gitignored); sizes go to results/gguf_<tag>.json.

Usage:
  python -m src.deploy.build_gguf --tag base --quants Q4_K_M
  python -m src.deploy.build_gguf --tag orpo_r16 --adapter <user>/finqa-llama32-3b-orpo-r16 --imatrix
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import src.paths  # noqa: F401  (HF cache on D:)
from src.data.render import CHAT_TEMPLATE_KWARGS

ROOT = Path(__file__).resolve().parents[2]
LLAMA_BIN = ROOT / "tools/llama.cpp/b11209/bin"
CONVERT = ROOT / "tools/llama.cpp-src/convert_hf_to_gguf.py"
CONVERT_PY = ROOT / "tools/.venv-convert/Scripts/python.exe"
BASE_FP16 = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]


def run(cmd: list, **kw) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def merged_hf_dir(tag: str, adapter: str | None) -> Path:
    from huggingface_hub import snapshot_download
    if not adapter:
        return Path(snapshot_download(BASE_FP16, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]))
    out = ROOT / "models" / tag / "hf"
    if not (out / "MERGE_OK.json").exists():
        merge_lora(adapter, out)
    return out


def merge_lora(adapter: str, out: Path) -> None:
    """Merge a LoRA adapter into the fp16 base by editing the safetensors directly: W' = W + (alpha/r) * B @ A,
    accumulated in fp32, stored as fp16.

    Done by hand because PeftModel.merge_and_unload() + save_pretrained() under transformers 5 wrote the *base*
    weights unchanged (verified: merged file identical to base, while the in-memory merge was correct).
    Config and tokenizer files are copied from the base (the tokenizer is untouched by LoRA, and transformers 5
    re-saving would write a tokenizer class the converter's transformers 4.57 can't load).
    """
    import shutil

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = Path(snapshot_download(BASE_FP16, allow_patterns=["*.safetensors", "*.json", "tokenizer*"]))
    if Path(adapter).exists():
        cfg_path, ad_path = Path(adapter) / "adapter_config.json", Path(adapter) / "adapter_model.safetensors"
    else:
        cfg_path = Path(hf_hub_download(adapter, "adapter_config.json"))
        ad_path = Path(hf_hub_download(adapter, "adapter_model.safetensors"))
    cfg = json.loads(cfg_path.read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    scaling = cfg["lora_alpha"] / cfg["r"]

    state = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                state[k] = s.get_tensor(k)
    merged = 0
    with safe_open(str(ad_path), "pt") as s:
        keys = set(s.keys())
        for a_key in sorted(k for k in keys if k.endswith(".lora_A.weight")):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            w_key = a_key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
            w = state[w_key]
            delta = s.get_tensor(b_key).float() @ s.get_tensor(a_key).float()
            state[w_key] = (w.float() + scaling * delta).to(w.dtype)
            merged += 1
    assert merged > 0, "no LoRA modules found in the adapter"

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.safetensors"):
        old.unlink()
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in base_dir.iterdir():
        if f.suffix == ".json" and not f.name.endswith(".index.json") or f.name.startswith("tokenizer"):
            shutil.copy(f, out / f.name)
    (out / "MERGE_OK.json").write_text(json.dumps({"adapter": adapter, "merged_modules": merged,
                                                    "scaling": scaling}, indent=2))


def calibration_file(path: Path, n: int = 200) -> Path:
    """Chat-formatted training examples (prompt + reference answer) for the importance matrix. Test data is
    never used, so the quantizer isn't tuned on what it is evaluated on."""
    if path.exists():
        return path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_FP16)
    rows = [json.loads(line) for line in (ROOT / "data/final/sft_train.jsonl").open(encoding="utf-8")][:n]
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, **CHAT_TEMPLATE_KWARGS) for r in rows]
    path.write_text("\n\n".join(texts), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", help="LoRA adapter (HF repo id or local dir); omit for the base model")
    ap.add_argument("--quants", default=",".join(DEFAULT_QUANTS))
    ap.add_argument("--imatrix", action="store_true")
    ap.add_argument("--imatrix-gpu-layers", type=int, default=14, help="layers offloaded while computing imatrix")
    args = ap.parse_args()

    out_dir = ROOT / "models" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{args.tag}-f16.gguf"
    if not f16.exists():
        hf_dir = merged_hf_dir(args.tag, args.adapter)
        run([CONVERT_PY, CONVERT, hf_dir, "--outtype", "f16", "--outfile", f16])

    files = {"f16": f16}
    for q in [q for q in args.quants.split(",") if q]:
        target = out_dir / f"{args.tag}-{q}.gguf"
        if not target.exists():
            run([LLAMA_BIN / "llama-quantize.exe", f16, target, q])
        files[q] = target

    if args.imatrix:
        imat = out_dir / "imatrix.gguf"
        if not imat.exists():
            calib = calibration_file(out_dir / "calibration_train.txt")
            run([LLAMA_BIN / "llama-imatrix.exe", "-m", f16, "-f", calib, "-o", imat, "-c", "2048",
                 "-ngl", str(args.imatrix_gpu_layers), "--chunks", "100"])
        target = out_dir / f"{args.tag}-Q4_K_M-imatrix.gguf"
        if not target.exists():
            run([LLAMA_BIN / "llama-quantize.exe", "--imatrix", imat, f16, target, "Q4_K_M"])
        files["Q4_K_M-imatrix"] = target

    report = {"tag": args.tag, "adapter": args.adapter, "llama_cpp_build": "b11209",
              "files": {k: {"path": str(v.relative_to(ROOT)), "size_gb": round(v.stat().st_size / 1024**3, 3)}
                        for k, v in files.items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/gguf_{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
