"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено."
_WITH_UNIT = re.compile(
    r"(?<![\w-])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)\s*(?:руб(?:\.|ля|лей)?|₽|шт(?:\.|ук(?:а|и)?)?)",
    re.IGNORECASE,
)


def _canonical_call_key(name: str, arguments: dict) -> str:
    normalized = dict(arguments)
    for field in ("part_id", "model"):
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = normalize_code(value) or value
    return json.dumps([name, normalized], ensure_ascii=False, sort_keys=True, default=str)


def _parts_in(value: object) -> set[str]:
    raw = json.dumps(value, ensure_ascii=False, default=str)
    return {code for code in find_codes(raw) if code_type(code) == "Part"}


def _quantity_is_in_question(quantity: object, question: str) -> bool:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        return False
    pattern = rf"(?<![\w-]){re.escape(str(quantity))}(?![\w-])"
    return re.search(pattern, question) is not None


def _grounded_numbers(results: list[dict]) -> set[str]:
    numbers: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"price_rub", "qty"} and isinstance(item, int) and not isinstance(item, bool):
                    numbers.add(str(item))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for result in results:
        collect(result)
    return numbers


def _grounding_errors(answer: str, allowed_codes: set[str], results: list[dict]) -> list[str]:
    errors = [code for code in find_codes(answer) if code not in allowed_codes]
    allowed_numbers = _grounded_numbers(results)
    for match in _WITH_UNIT.finditer(answer):
        number = re.sub(r"\D", "", match.group(1))
        if number not in allowed_numbers:
            errors.append(match.group(0))
    return errors


def _interrupted(steps: list[ToolStep], sources: set[str]) -> AgentResult:
    known = [step.result for step in steps if step.result is not None]
    suffix = f" Полученные данные: {json.dumps(known, ensure_ascii=False)}" if known else ""
    return AgentResult(f"Работа прервана: достигнут лимит шагов.{suffix}", steps, sorted(sources))


def _record_error(call: ToolCall, arguments: dict, error: str, steps: list[ToolStep], messages: list[dict]) -> None:
    steps.append(ToolStep(call.name, arguments, error=error))
    messages.append(tool_message(call, {"error": error}))


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Ты ассистент сервисной службы. Используй инструменты для фактов и ERP. "
                "Не придумывай номера деталей, цены и остатки. Заказ предлагай только с явными аргументами."
            ),
        },
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    results: list[dict] = []
    cached_calls: dict[str, dict] = {}
    allowed_codes = set(find_codes(question))
    allowed_parts = {code for code in allowed_codes if code_type(code) == "Part"}
    knowledge_refused = False
    missing_data = False

    while True:
        if len(steps) >= max_steps:
            return _interrupted(steps, sources)

        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not response.tool_calls:
            answer = response.content.strip()
            if not answer:
                return AgentResult(REFUSAL, steps, sorted(sources), refused=True)

            errors = _grounding_errors(answer, allowed_codes, results)
            if errors:
                messages.extend(
                    [
                        response.as_message(),
                        {
                            "role": "user",
                            "content": (
                                "Исправь ответ: используй только коды из вопроса и результатов инструментов, "
                                "а цены и остатки — только из ERP. Не повторяй неподтверждённые факты."
                            ),
                        },
                    ]
                )
                corrected = llm.chat(messages)
                answer = corrected.content.strip()
                if corrected.tool_calls or not answer or _grounding_errors(answer, allowed_codes, results):
                    return AgentResult(REFUSAL, steps, sorted(sources), refused=True)

            refused = knowledge_refused or missing_data or REFUSAL.lower() in answer.lower()
            return AgentResult(answer, steps, sorted(sources), refused=refused)

        messages.append(response.as_message())
        for call in response.tool_calls:
            if len(steps) >= max_steps:
                return _interrupted(steps, sources)

            arguments = call.arguments if isinstance(call.arguments, dict) else {}
            if call.name not in tools:
                _record_error(call, arguments, f"неизвестный инструмент: {call.name}", steps, messages)
                continue
            if "_raw" in arguments:
                _record_error(call, arguments, "аргументы инструмента не являются JSON", steps, messages)
                continue

            part_id = arguments.get("part_id")
            if part_id is not None:
                found_parts = _parts_in(part_id)
                if len(found_parts) != 1 or not found_parts <= allowed_parts:
                    _record_error(
                        call,
                        arguments,
                        "номер детали не подтверждён; сначала вызови search_knowledge",
                        steps,
                        messages,
                    )
                    continue

            if call.name == "create_service_order" and not _quantity_is_in_question(arguments.get("qty"), question):
                _record_error(call, arguments, "количество заказа не названо пользователем", steps, messages)
                continue

            if call.name == "create_service_order" and not confirm:
                error = "создание заказа ожидает подтверждения пользователя"
                steps.append(ToolStep(call.name, arguments, error=error))
                return AgentResult(
                    "Подтвердите создание заказа.",
                    steps,
                    sorted(sources),
                    needs_confirmation=True,
                    pending_action={"tool": call.name, "args": arguments},
                )

            cache_key = _canonical_call_key(call.name, arguments)
            if cache_key in cached_calls:
                result = cached_calls[cache_key]
            else:
                result = None
                unavailable_error = None
                for _ in range(3):
                    try:
                        result = tools[call.name](**arguments)
                        break
                    except ErpUnavailable as exc:
                        unavailable_error = str(exc)
                    except TypeError as exc:
                        result = {"error": f"некорректные аргументы: {exc}"}
                        break
                if result is None:
                    result = {"error": unavailable_error or "Данные ERP получить не удалось."}
                cached_calls[cache_key] = result

            error = result.get("error") if isinstance(result, dict) else "инструмент вернул некорректный результат"
            if not isinstance(result, dict):
                result = {"error": error}
            steps.append(ToolStep(call.name, arguments, result=result, error=error))
            results.append(result)
            messages.append(tool_message(call, result))

            if error:
                missing_data = missing_data or str(error).lower().startswith("данных нет")
            else:
                source_list = result.get("sources", [])
                if isinstance(source_list, list):
                    sources.update(source for source in source_list if isinstance(source, str))
                found_codes = set(find_codes(json.dumps(result, ensure_ascii=False, default=str)))
                allowed_codes.update(found_codes)
                allowed_parts.update(code for code in found_codes if code_type(code) == "Part")
                knowledge_refused = knowledge_refused or bool(result.get("refused"))
