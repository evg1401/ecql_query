from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

RULES_XLSX = "datasets/ecql_dataset_rules.xlsx"
ADAPTER_DIR = Path("outputs/ecql-lora")

MAX_SEQ_LEN = 2048
VALID_EVERY = 10    # валидировать каждый {VALID_EVERY} пример
SEED = 42

EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

SYSTEM_HEADER = (
    "Ты переводишь запросы с русского языка в язык запросов ECQL.\n"
    "Отвечай ровно одной строкой — готовым запросом ECQL. "
    "Без пояснений, без блоков кода, без точки в конце."
)

def compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
        llm_int8_skip_modules=["lm_head"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
        dtype=compute_dtype(),
    )

    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    ))
    model.print_trainable_parameters()

    model.config.use_cache = False
    return model, tokenizer


# данные из книги правил
def read_rules(path: str = RULES_XLSX):
    import pandas as pd

    book = pd.read_excel(path, sheet_name=None)
    pairs = book["Датасет"].dropna(subset=["Запрос на русском языке"]).to_dict("records")

    lines = [SYSTEM_HEADER, "", "# Правила ECQL"]
    for row in book["Соглашения"].to_dict("records"):
        lines.append(f"- {row['Правило']}: {row['Описание']}")

    codes: dict[tuple[str, str], list[str]] = {}
    for row in book["Справочники"].to_dict("records"):
        codes.setdefault((row["Сущность"], row["Поле"]), []).append(str(row["Код в ECQL"]))

    lines += ["", "# Схема: сущности и поля"]
    schema = book["Схема"].to_dict("records")
    for entity in dict.fromkeys(r["Сущность"] for r in schema):
        lines.append(f"[{entity}]")
        plain: dict[str, list[str]] = {}
        for row in (r for r in schema if r["Сущность"] == entity):
            allowed = codes.get((entity, row["Поле"]))
            if allowed:
                lines.append(f"  {row['Поле']}: {'|'.join(allowed)}")
                continue

            kind = str(row["Тип значения"])
            group = "даты" if kind.startswith("дата") else "числа" if kind == "число" else "строки"
            plain.setdefault(group, []).append(str(row["Поле"]))
        for group in ("строки", "числа", "даты"):
            if group in plain:
                lines.append(f"  {group}: {' '.join(plain[group])}")

    return pairs, "\n".join(lines)


def render_prompt(tokenizer, system_prompt: str, query: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": str(query).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode(tokenizer, system_prompt: str, query: str, answer: str) -> dict:
    prompt = tokenizer(render_prompt(tokenizer, system_prompt, query),
                       add_special_tokens=False)["input_ids"]
    target = tokenizer(str(answer).strip() + tokenizer.eos_token,
                       add_special_tokens=False)["input_ids"]
    overflow = len(prompt) + len(target) - MAX_SEQ_LEN
    if overflow > 0:
        prompt = prompt[overflow:]
    return {"input_ids": prompt + target,
            "labels": [-100] * len(prompt) + target,
            "attention_mask": [1] * (len(prompt) + len(target))}


def split_pairs(pairs) -> tuple[list[dict], list[dict]]:
    # разбиение на train/valid
    order = {"Простой": 0, "Средний": 1, "Сложный": 2}
    pairs = sorted(pairs, key=lambda r: order.get(str(r.get("Сложность")), 3))
    valid = pairs[::VALID_EVERY]
    train = [p for i, p in enumerate(pairs) if i % VALID_EVERY]
    random.Random(SEED).shuffle(train)
    return train, valid


def encode_all(tokenizer, system_prompt: str, rows) -> list[dict]:
    return [encode(tokenizer, system_prompt,
                   r["Запрос на русском языке"], r["Запрос на языке ECQL"])
            for r in rows]


def collate(features, pad_id: int):
    width = max(len(f["input_ids"]) for f in features)
    pad = lambda seq, value: seq + [value] * (width - len(seq))
    return {
        "input_ids": torch.tensor([pad(f["input_ids"], pad_id) for f in features]),
        "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
        "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
    }

# построение кривой обучения
def plot_loss(history: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
    valid = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if not train:
        raise ValueError("в истории нет записей с loss")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*train), marker=".", label=f"train ({len(train)} точек)")
    if valid:
        ax.plot(*zip(*valid), marker="o", label=f"eval ({len(valid)})")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title(f"r={LORA_R}, alpha={LORA_ALPHA}, lr={LEARNING_RATE}, эпох {EPOCHS}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def train() -> None:
    pairs, system_prompt = read_rules()
    model, tokenizer = load_model()
    train_rows, valid_rows = split_pairs(pairs)
    train_ds = encode_all(tokenizer, system_prompt, train_rows)
    valid_ds = encode_all(tokenizer, system_prompt, valid_rows)
    print(f"пар: {len(pairs)} | train/valid: {len(train_ds)}/{len(valid_ds)} | "
          f"промпт: {len(tokenizer(system_prompt)['input_ids'])} токенов | "
          f"самый длинный пример: {max(len(e['input_ids']) for e in train_ds)} из {MAX_SEQ_LEN}")

    per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = per_epoch * EPOCHS
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))
    print(f"шагов оптимизатора: {total_steps} (разогрев {warmup_steps})")

    use_bf16 = compute_dtype() is torch.bfloat16
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="outputs/run",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            max_grad_norm=0.3,
            optim="paged_adamw_8bit",
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            remove_unused_columns=False,
            report_to=[],
            seed=SEED,
        ),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=lambda f: collate(f, tokenizer.pad_token_id),
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    (ADAPTER_DIR / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    print(f"\nАдаптер сохранён: {ADAPTER_DIR}")

    history = trainer.state.log_history
    (ADAPTER_DIR / "log_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        plot_loss(history, ADAPTER_DIR / "loss_curve.png")
        print(f"График обучения: {ADAPTER_DIR / 'loss_curve.png'}")
    except Exception as exc:
        print(f"График не построен ({type(exc).__name__}: {exc}); "
              f"история в {ADAPTER_DIR / 'log_history.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="дообучить на книге правил")
    mode.add_argument("--prompt", action="store_true", help="показать системный промпт")
    args = p.parse_args()

    if args.train:
        train()
    elif args.prompt:
        print(read_rules()[1])


if __name__ == "__main__":
    main()
