from __future__ import annotations

import os
from pathlib import Path


MODEL_NAME = "Qwen2.5-3B-Instruct"


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _resolve_snapshot(path: Path) -> Path | None:
    if _is_model_dir(path):
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        matches = sorted(p.parent for p in snapshots.glob("*/config.json"))
        if matches:
            return matches[-1]
    return None


def resolve_model_path(value: str | Path | None) -> Path:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value).expanduser())
    if os.environ.get("QWEN_MODEL_PATH"):
        candidates.append(Path(os.environ["QWEN_MODEL_PATH"]).expanduser())

    cwd = Path.cwd()
    for base in (cwd, cwd.parent, Path.home(), Path("C:/Users/Public")):
        candidates.extend(
            [
                base / MODEL_NAME,
                base / "models" / MODEL_NAME,
                base / "shared" / MODEL_NAME,
                base / "공유" / MODEL_NAME,
                base / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-3B-Instruct",
            ]
        )

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        resolved = _resolve_snapshot(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Qwen2.5-3B-Instruct를 찾지 못했습니다. --model-path 또는 "
        "QWEN_MODEL_PATH로 config.json이 있는 폴더를 지정하세요.\n확인한 경로:\n- "
        + "\n- ".join(checked)
    )


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=False, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: Path,
    adapter_path: str | Path | None = None,
    quantization: str = "bf16",
    for_training: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. Qwen 3B 실행에는 CUDA 환경을 권장합니다.")

    kwargs = {
        "device_map": "auto",
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization != "bf16":
        raise ValueError(f"unsupported quantization mode: {quantization}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=for_training)
    model.config.use_cache = not for_training
    return model
