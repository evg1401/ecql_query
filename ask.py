from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import ecql

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = Path("outputs/ecql-lora")
DOCUMENT = "datasets/ecql_shop_test.xlsx"

LIMIT = 20
MAX_NEW_TOKENS = 96
CPU_DTYPE = torch.float32



# загрузка модели
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    build = "сборка torch" if "+cpu" in torch.__version__ else "GPU не найден"
    print(f"[инференс] CUDA недоступна ({build}) -> CPU")
    return "cpu"


def load_model(device: str | None = None):
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise SystemExit(
            f"Нет адаптера в {ADAPTER_DIR} — сначала обучите: python ecql_lora.py --train"
        )
    device = device or pick_device()

    kwargs: dict = {"attn_implementation": "sdpa"}
    if device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            llm_int8_skip_modules=["lm_head"],
        )
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = kwargs["quantization_config"].bnb_4bit_compute_dtype
        print(f"[инференс] CUDA: {torch.cuda.get_device_name(0)}, 4-битная NF4")
    else:
        kwargs["device_map"] = "cpu"
        kwargs["dtype"] = CPU_DTYPE
        gib = {"float32": 16, "bfloat16": 8, "float16": 8}.get(
            str(CPU_DTYPE).split(".")[-1], 16
        )
        print(f"[инференс] CPU, {str(CPU_DTYPE).split('.')[-1]} без квантования "
              f"(~{gib} ГБ RAM, ~1 мин на запрос)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.config.use_cache = True
    model.eval()

    system_prompt = (ADAPTER_DIR / "system_prompt.txt").read_text(encoding="utf-8")
    return model, tokenizer, system_prompt

@torch.inference_mode()
def to_ecql(model, tokenizer, system_prompt: str, question: str) -> str:
    """Переводит вопрос на естественном языке в запрос ECQL."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = out[0][enc["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# исполняет запрос ecql по документу
def load_from_document(query: str, tables: dict | None = None,
                       document: str = DOCUMENT) -> tuple[list[dict], str | None, int]:
    tables = tables if tables is not None else ecql.load_tables(document)
    return ecql.execute(query, tables)
