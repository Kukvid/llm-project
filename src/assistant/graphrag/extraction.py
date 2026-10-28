"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

EXTRACTION_PROMPT = """Извлеки из текста только сущности и направленные связи заданной схемы.

Допустимые направления:
Model -BELONGS_TO-> ProductLine
Model -HAS_PART-> Part
Model -FAILED-> Component
Part -IS_A-> Component
Part -REPLACED_BY-> Part
Part -SUPPLIED_BY-> Supplier
ErrorCode -INDICATES-> Component
ErrorCode -FIXED_BY-> Procedure

Поле subject — источник стрелки слева, поле object — цель стрелки справа.
Не добавляй факты, которых нет в тексте. Пример: из фразы «Модель AR-210 входит в
линейку Арктика» извлеки Model AR-210, ProductLine Арктика и связь
{"subject_type":"Model","subject":"AR-210","type":"BELONGS_TO",
"object_type":"ProductLine","object":"Арктика"}. Для фразы «деталь P-1010 — компрессор»
связь IS_A имеет subject Part P-1010 и object Component компрессор, а не наоборот.
Верни только JSON по переданной схеме.
"""


def _parse_response(content: str) -> tuple[list[dict], list[dict]]:
    data = json.loads(content)
    if not isinstance(data, dict) or not {"entities", "relations"} <= set(data):
        raise ValueError("ответ не содержит entities и relations")
    entities, relations = data["entities"], data["relations"]
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError("entities и relations должны быть списками")

    for item in entities:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("type"), str)
            or not isinstance(item.get("name"), str)
            or item["type"] not in ENTITY_TYPES
        ):
            raise ValueError("некорректная сущность")
    required = {"subject_type", "subject", "type", "object_type", "object"}
    for item in relations:
        if (
            not isinstance(item, dict)
            or not required <= set(item)
            or not all(isinstance(item[key], str) for key in required)
            or item["subject_type"] not in ENTITY_TYPES
            or item["object_type"] not in ENTITY_TYPES
            or item["type"] not in RELATION_SIGNATURES
        ):
            raise ValueError("некорректная связь")
    return entities, relations


def _canonical_entity(
    entity_type: str,
    raw_name: str,
    names: dict[str, list[str]],
    codes_in_chunk: set[str],
) -> Entity | None:
    name = canonical_name(entity_type, raw_name, names)
    if name is None or (entity_type in CODE_TYPES and name not in codes_in_chunk):
        return None
    return Entity(entity_type, name)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": chunk.text},
    ]
    parsed = None
    for attempt in range(2):
        attempt_messages = messages
        if attempt:
            attempt_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Предыдущий ответ был обрезан или не прошёл проверку. Сформируй валидный JSON заново. "
                        "В каждой связи subject — источник стрелки, object — её цель. Для IS_A источником всегда "
                        "служит Part, а целью Component."
                    ),
                },
            ]
        response = llm.chat(
            attempt_messages,
            json_schema=EXTRACTION_SCHEMA,
            max_tokens=1024 if attempt == 0 else 2048,
        )
        try:
            parsed = _parse_response(response.content)
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if parsed is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    raw_entities, raw_relations = parsed
    names = vocabulary()
    codes_in_chunk = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    dropped = 0

    for item in raw_entities:
        entity = _canonical_entity(item["type"], item["name"], names, codes_in_chunk)
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    for item in raw_relations:
        subject = _canonical_entity(item["subject_type"], item["subject"], names, codes_in_chunk)
        object_ = _canonical_entity(item["object_type"], item["object"], names, codes_in_chunk)
        if subject is None or object_ is None:
            dropped += 1
            continue
        relation = Relation(subject, item["type"], object_)
        if not relation.matches_schema():
            dropped += 1
            continue
        entities[subject.node_id] = subject
        entities[object_.node_id] = object_
        relations[(subject.node_id, relation.type, object_.node_id)] = relation

    return ExtractionResult(
        chunk.chunk_id,
        chunk.doc_id,
        sorted(entities.values(), key=lambda entity: entity.node_id),
        [relations[key] for key in sorted(relations)],
        dropped,
    )
