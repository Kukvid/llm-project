"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Найти конкретные факты и цепочки в графе знаний.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Ответить на обзорный вопрос по сводкам сообществ.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на складе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить цену детали в рублях.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string"}},
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать подтверждённый заказ детали для сервисной заявки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def _part_id(raw: object) -> str | None:
    code = normalize_code(raw) if isinstance(raw, str) else None
    return code if code and code_type(code) == "Part" else None


def _model_id(raw: object) -> str | None:
    code = normalize_code(raw) if isinstance(raw, str) else None
    return code if code and code_type(code) == "Model" else None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "вопрос должен быть непустой строкой"}
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "вопрос должен быть непустой строкой"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        canonical = _part_id(part_id)
        if canonical is None:
            return {"error": "некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "неизвестный склад"}
        try:
            return erp.get_stock(canonical, warehouse)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"запрос отклонён ERP: {exc}"}

    def get_price(part_id: str) -> dict:
        canonical = _part_id(part_id)
        if canonical is None:
            return {"error": "некорректный номер детали"}
        try:
            return erp.get_price(canonical)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"запрос отклонён ERP: {exc}"}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        canonical_part = _part_id(part_id)
        canonical_model = _model_id(model)
        if canonical_part is None:
            return {"error": "некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "неизвестный склад"}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "количество должно быть положительным целым числом"}
        if canonical_model is None:
            return {"error": "некорректная модель"}
        try:
            return erp.create_order(canonical_part, warehouse, qty, canonical_model)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"запрос отклонён ERP: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
