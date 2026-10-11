from __future__ import annotations
import re
from typing import Any

FORMATS = ("JSON", "TABLE", "LIST")

QUERY = re.compile(r"^FETCH\s+\[([A-Z_]+)\]((?:\s+.*?)??)(?:\s+AS\s+([A-Z]+))?$")
COND = re.compile(r"^@([a-z_]+)\s+(IS|NOT|ABOVE|BELOW)\s+(\"[^\"]*\"|-?\d+(?:\.\d+)?)$")

Condition = tuple[str, str, str]


def normalize(query: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*|```$", "", query.strip()).strip().strip("`")
    text = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return re.sub(r"\s+", " ", text.rstrip(".;")).strip()


def parse(query: str) -> tuple[str, list[list[Condition]], str | None]:
    text = normalize(query)
    m = QUERY.match(text)
    if not m:
        raise ValueError(f"не соответствует шаблону FETCH [СУЩНОСТЬ] ...: {text!r}")
    entity, raw, fmt = m.group(1), (m.group(2) or "").strip(), m.group(3)
    if fmt and fmt not in FORMATS:
        raise ValueError(f"неизвестный формат вывода AS {fmt}")

    groups: list[list[Condition]] = []
    if raw:
        for raw_group in re.split(r"\s+\|\|\s+", raw):
            conditions: list[Condition] = []
            for raw_cond in re.split(r"\s+&&\s+", raw_group.strip()):
                cm = COND.match(raw_cond.strip())
                if not cm:
                    raise ValueError(f"не разобрано условие: {raw_cond.strip()!r}")
                conditions.append(cm.groups())  # type: ignore[arg-type]
            groups.append(conditions)
    return entity, groups, fmt


# исполнение на таблицах
def load_tables(path: str) -> dict[str, list[dict]]:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    return {
        str(name).strip(): df.dropna(how="all").to_dict("records")
        for name, df in sheets.items()
        if str(name).strip() != "Описание"
    }


def blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and value != value)
        or (isinstance(value, str) and not value.strip())
    )


def match(cell: Any, op: str, raw: str) -> bool:
    wanted: Any = raw[1:-1] if raw.startswith('"') else float(raw)
    if blank(cell):
        return op == "NOT"
    try:
        left, right = float(cell), float(wanted)
    except (TypeError, ValueError):
        left, right = str(cell).strip(), str(wanted).strip()
    if op == "IS":
        return left == right
    if op == "NOT":
        return left != right
    if op == "ABOVE":
        return left > right
    return left < right


def execute(query: str, tables: dict[str, list[dict]]) -> tuple[list[dict], str | None, int]:
    entity, groups, fmt = parse(query)
    if entity not in tables:
        raise ValueError(f"нет данных для [{entity}]; есть: {', '.join(sorted(tables))}")
    rows = tables[entity]
    hits = [
        row for row in rows
        if not groups or any(all(match(row.get(f), op, v) for f, op, v in g) for g in groups)
    ]
    return hits, fmt, len(rows)

