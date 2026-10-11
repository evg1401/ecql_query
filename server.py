"""HTTP-сервер над пайплайном из ask.py.

Модель загружается один раз при старте, дальше каждый запрос только генерирует.
На CPU это главный выигрыш: 18 с загрузки платятся однократно, а не на каждый
вопрос (замер — в README).

    python server.py

    GET  /health                   устройство, dtype, таблицы документа
    GET  /ask?q=<вопрос>&limit=20  вопрос -> ECQL -> строки документа

Параметр limit: сколько строк вернуть, 0 — все. По умолчанию ask.LIMIT.
Вопрос в URL должен быть закодирован в UTF-8 (браузер делает это сам).

Используются только стандартная библиотека и уже имеющиеся модули проекта.
Сервер слушает localhost: наружу не выставляется, авторизации нет.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import ask
import ecql

HOST = "127.0.0.1"
PORT = 8000

MODEL = TOKENIZER = SYSTEM_PROMPT = TABLES = None
MODEL_LOCK = threading.Lock()


def jsonable(rows: list[dict]) -> list[dict]:
    """Готовит строки к сериализации.

    В ячейках xlsx попадаются NaN; json.dumps выдал бы для них литерал NaN,
    который не является валидным JSON. Такие значения становятся null.
    """
    clean = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                item[str(key)] = None
            elif isinstance(value, (str, int, float, bool)) or value is None:
                item[str(key)] = value
            else:
                item[str(key)] = str(value)
        clean.append(item)
    return clean


def execute(query: str, limit: int) -> dict:
    """Вторая половина пайплайна: исполняет запрос ECQL по документу."""
    rows, fmt, scanned = ask.load_from_document(query, TABLES)
    shown = rows[:limit] if limit else rows
    return {
        "ecql": ecql.normalize(query),
        "found": len(rows),
        "scanned": scanned,
        "format": fmt,
        "returned": len(shown),
        "rows": jsonable(shown),
    }


def answer(question: str, limit: int) -> dict:
    """Полный пайплайн: вопрос -> ECQL -> строки документа."""
    with MODEL_LOCK:
        query = ask.to_ecql(MODEL, TOKENIZER, SYSTEM_PROMPT, question)
    print(f"  вопрос: {question}")
    print(f"  ECQL  : {query}")
    result = {"question": question}
    result.update(execute(query, limit))
    print(f"  найдено {result['found']} из {result['scanned']}")
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "ecql-lora"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.address_string()}] {fmt % args}")

    # ответы
    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    @staticmethod
    def pick_limit(value) -> int:
        if value is None or value == "":
            return ask.LIMIT
        try:
            limit = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"limit должен быть целым, получено {value!r}")
        if limit < 0:
            raise ValueError("limit не может быть отрицательным")
        return limit

    # маршруты
    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}

        if url.path == "/health":
            self.send_json({
                "device": str(MODEL.device),
                "dtype": str(next(MODEL.parameters()).dtype),
                "model": ask.MODEL_ID,
                "adapter": str(ask.ADAPTER_DIR),
                "document": ask.DOCUMENT,
                "tables": {name: len(rows) for name, rows in sorted(TABLES.items())},
            })
        elif url.path == "/ask":
            self.handle_ask(params.get("q"), params.get("limit"))
        else:
            self.fail(f"нет маршрута {url.path}; есть /ask и /health", 404)

    def handle_ask(self, text, raw_limit) -> None:
        if not text or not str(text).strip():
            return self.fail("нужен непустой параметр q")
        try:
            limit = self.pick_limit(raw_limit)
        except ValueError as exc:
            return self.fail(str(exc))

        try:
            self.send_json(answer(str(text).strip(), limit))
        except ValueError as exc:
            self.fail(str(exc), 422)
        except Exception as exc:
            self.fail(f"{type(exc).__name__}: {exc}", 500)


def main() -> None:
    global MODEL, TOKENIZER, SYSTEM_PROMPT, TABLES

    TABLES = ecql.load_tables(ask.DOCUMENT)
    print(f"Документ: {ask.DOCUMENT}")
    print("Таблицы : " + ", ".join(f"{e} ({len(r)})" for e, r in sorted(TABLES.items())))

    MODEL, TOKENIZER, SYSTEM_PROMPT = ask.load_model()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nСлушаю http://{HOST}:{PORT}")
    print(f"  GET http://{HOST}:{PORT}/health")
    print(f"  GET http://{HOST}:{PORT}/ask?q=<вопрос>")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nсервер остановлен")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
