"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from collections import Counter
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, KnowledgeBase

REFUSAL = "В базе знаний не найдено."
MAX_FACTS_LENGTH = 8000
EMPTY_PARTIAL_ANSWERS = {
    "",
    "информация отсутствует",
    "не указано",
    "нет информации",
    "нельзя определить",
    "невозможно определить",
    REFUSAL.casefold().rstrip("."),
}


def _add_pair_edges(graph: nx.DiGraph, selected: set[tuple[str, str]], left: str, right: str) -> None:
    if graph.has_edge(left, right):
        selected.add((left, right))
    if graph.has_edge(right, left):
        selected.add((right, left))


def _replacement_facts(graph: nx.DiGraph, starts: set[str]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    pending = sorted(starts)
    visited: set[str] = set()
    while pending:
        part = pending.pop(0)
        if part in visited:
            continue
        visited.add(part)
        successors = sorted(graph.successors(part))
        replacements = [target for target in successors if graph.edges[part, target].get("type") == "REPLACED_BY"]
        for target in replacements:
            selected.add((part, target))
            pending.append(target)
        if not replacements:
            for target in successors:
                if graph.edges[part, target].get("type") != "SUPPLIED_BY":
                    continue
                selected.add((part, target))
    return selected


def _document_sources(raw_sources: list[str]) -> set[str]:
    return {source.split("#", 1)[0] for source in raw_sources}


def _refusal() -> Answer:
    return Answer(REFUSAL, refused=True)


def _community_facts(community, graph: nx.DiGraph) -> str:
    nodes = set(community.nodes)
    facts: list[str] = []
    length = 0
    for source, target, attrs in sorted(graph.edges(data=True)):
        if source not in nodes and target not in nodes:
            continue
        document_count = len(_document_sources(attrs.get("sources", [])))
        fact = (
            f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}; "
            f"вес: {attrs.get('weight', document_count)}; источников: {document_count}"
        )
        extra = len(fact) + (1 if facts else 0)
        if length + extra > MAX_FACTS_LENGTH:
            break
        facts.append(fact)
        length += extra
    return "\n".join(f"- {fact}" for fact in facts)


def _is_empty_partial(answer: str) -> bool:
    normalized = answer.strip().casefold().rstrip(".!?")
    return normalized in EMPTY_PARTIAL_ANSWERS


def _report_aggregates(graph: nx.DiGraph) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, set[str]]]]:
    """Считает обзорные данные по FAILED и REPLACED_BY вместе с их источниками."""
    model_lines = {
        source: (graph.nodes[target]["name"], _document_sources(attrs.get("sources", [])))
        for source, target, attrs in graph.edges(data=True)
        if attrs.get("type") == "BELONGS_TO"
    }
    component_counts: Counter = Counter()
    model_counts: Counter = Counter()
    sources_by_line: dict[str, dict[str, set[str]]] = {}
    for model, component, attrs in graph.edges(data=True):
        if attrs.get("type") != "FAILED" or model not in model_lines:
            continue
        line, membership_sources = model_lines[model]
        failure_sources = _document_sources(attrs.get("sources", []))
        count = int(attrs.get("weight") or len(failure_sources))
        component_counts[(line, graph.nodes[component]["name"])] += count
        model_counts[(line, graph.nodes[model]["name"])] += count
        sources_by_line.setdefault(line, {}).setdefault("failure", set()).update(
            membership_sources | failure_sources
        )

    failure_lines: dict[str, list[str]] = {}
    for (line, component), count in sorted(component_counts.items()):
        failure_lines.setdefault(line, []).append(f"Линейка {line}; узел {component}; отказов: {count}.")
    for (line, model), count in sorted(model_counts.items()):
        failure_lines.setdefault(line, []).append(f"Линейка {line}; модель {model}; всего отказов: {count}.")
    for line, values in failure_lines.items():
        components = [
            (component, count)
            for (item_line, component), count in component_counts.items()
            if item_line == line
        ]
        models = [(model, count) for (item_line, model), count in model_counts.items() if item_line == line]
        if components:
            maximum = max(count for _, count in components)
            leaders = ", ".join(sorted(component for component, count in components if count == maximum))
            values.append(f"Максимум отказов среди узлов: {leaders}; отказов: {maximum}.")
        if models:
            maximum = max(count for _, count in models)
            leaders = ", ".join(sorted(model for model, count in models if count == maximum))
            values.append(f"Максимум отказов среди моделей: {leaders}; отказов: {maximum}.")

    parts_by_line: dict[str, dict[str, set[str]]] = {}
    for model, (line, membership_sources) in model_lines.items():
        for _, part, attrs in graph.out_edges(model, data=True):
            if attrs.get("type") == "HAS_PART":
                part_sources = _document_sources(attrs.get("sources", [])) | membership_sources
                parts_by_line.setdefault(line, {}).setdefault(part, set()).update(part_sources)
    replacement_lines: dict[str, list[str]] = {}
    for line, installed_parts in parts_by_line.items():
        discontinued: set[str] = set()
        for old_part, _new_part, attrs in graph.edges(data=True):
            if attrs.get("type") != "REPLACED_BY" or old_part not in installed_parts:
                continue
            discontinued.add(old_part)
            replacement_sources = _document_sources(attrs.get("sources", []))
            sources_by_line.setdefault(line, {}).setdefault("replacement", set()).update(
                installed_parts[old_part] | replacement_sources
            )
        if discontinued:
            names = ", ".join(sorted(graph.nodes[part]["name"] for part in discontinued))
            replacement_lines.setdefault(line, []).append(
                f"Снятые с производства детали из руководств: {names}."
            )
    texts: dict[str, dict[str, str]] = {}
    for line, values in failure_lines.items():
        texts.setdefault(line, {})["failure"] = "\n".join(values)
    for line, values in replacement_lines.items():
        texts.setdefault(line, {})["replacement"] = "\n".join(values)
    return texts, sources_by_line


def _aggregate_intent(question: str) -> str | None:
    normalized = question.casefold()
    if "снят" in normalized and "производств" in normalized:
        return "replacement"
    if "отказ" in normalized and any(
        marker in normalized for marker in ("чаще", "част", "больше", "наибольш", "максим")
    ):
        return "failure"
    return None


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = find_codes(question)
    code_nodes = [f"{code_type(code)}:{code}" for code in codes]
    if any(node not in graph for node in code_nodes):
        return _refusal()

    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = list(dict.fromkeys([*code_nodes, *(node for node in name_nodes if node in graph)]))
    if not anchors:
        return _refusal()

    undirected = graph.to_undirected()
    selected: set[tuple[str, str]] = set()
    if len(anchors) == 1:
        neighbourhood = set(nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2))
        for source, target in graph.edges:
            if source in neighbourhood and target in neighbourhood:
                selected.add((source, target))
    else:
        path_nodes: set[str] = set()
        for left, right in combinations(anchors, 2):
            if not nx.has_path(undirected, left, right):
                continue
            for path in nx.all_simple_paths(undirected, left, right, cutoff=3):
                path_nodes.update(path)
                for source, target in zip(path, path[1:], strict=False):
                    _add_pair_edges(graph, selected, source, target)

        parts = {node for node in path_nodes if graph.nodes[node].get("type") == "Part"}
        selected.update(_replacement_facts(graph, parts))

    if not selected:
        return _refusal()

    facts: list[str] = []
    sources: set[str] = set()
    length = 0
    for source, target in sorted(selected):
        attrs = graph.edges[source, target]
        fact = f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}"
        numbered = f"{len(facts) + 1}. {fact}"
        extra = len(numbered) + (1 if facts else 0)
        if length + extra > MAX_FACTS_LENGTH:
            break
        facts.append(fact)
        length += extra
        sources.update(_document_sources(attrs.get("sources", [])))

    if not facts:
        return _refusal()

    fact_list = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    prompt = (
        "Ответь на вопрос только по пронумерованным фактам. Не придумывай коды и связи. "
        "Если есть цепочка REPLACED_BY, называй последнюю деталь цепочки.\n\n"
        f"Вопрос: {question}\n\nФакты:\n{fact_list}"
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    if not response.content.strip():
        return _refusal()
    return Answer(response.content, sorted(sources), facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    code_nodes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in kb.graph for node in code_nodes):
        return _refusal()

    aggregates, aggregate_sources = _report_aggregates(kb.graph)
    question_lines = {
        name for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"
    }
    aggregate_intent = _aggregate_intent(question)
    requested_statistics = ""
    if aggregate_intent:
        requested_statistics = "\n".join(
            aggregates[line][aggregate_intent]
            for line in sorted(question_lines)
            if aggregate_intent in aggregates.get(line, {})
        )
    candidates: list[tuple[str, Community]] = []
    for community in kb.communities:
        facts = _community_facts(community, kb.graph)
        community_statistics = ""
        if aggregate_intent:
            community_statistics = "\n".join(
                aggregates[kb.graph.nodes[node]["name"]][aggregate_intent]
                for node in community.nodes
                if node in kb.graph
                and kb.graph.nodes[node].get("type") == "ProductLine"
                and aggregate_intent in aggregates.get(kb.graph.nodes[node]["name"], {})
            )
        statistics = requested_statistics or community_statistics
        prompt = (
            "Ответь на вопрос только по сводке сообщества. Верни JSON с полями answer и score, "
            "где score от 0 до 10 показывает полезность данных. Если прямого ответа или данных "
            "для подсчёта нет, верни пустой answer и score 0. Не выдавай фразу об отсутствии "
            "информации за ответ. Поле «документов» — число независимых документов, "
            "подтверждающих связь. Проверенная статистика приоритетнее сводки: для вопроса "
            "«чаще всего» выбери запись с максимальным числом.\n\n"
            f"Вопрос: {question}\nСводка: {community.summary}\nПроверяемые факты:\n{facts or '- нет'}\n"
            f"Проверенная статистика:\n{statistics or '- нет'}"
        )
        response = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA)
        try:
            data = json.loads(response.content)
            answer, score = data["answer"], data["score"]
            valid_score = not isinstance(score, bool) and isinstance(score, int) and 0 <= score <= 10
            if not isinstance(answer, str) or not valid_score:
                continue
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if score > 0 and not _is_empty_partial(answer):
            candidates.append((answer, community))

    partials = [answer for answer, _ in candidates]
    useful_communities = [community for _, community in candidates]

    if not partials:
        return _refusal()

    joined = "\n".join(f"{number}. {answer}" for number, answer in enumerate(partials, start=1))
    prompt = (
        "Объедини частичные ответы в один краткий ответ на вопрос. Используй только приведённые данные. "
        "Для вопросов о частоте сравни указанные количества, а не число упоминаний в тексте ответа. "
        "Не суммируй одинаковые частичные ответы и не придумывай новые числа. Проверенная статистика "
        "приоритетнее частичных ответов.\n\n"
        f"Вопрос: {question}\n\nПроверенная статистика:\n{requested_statistics or '- нет'}\n\n"
        f"Частичные ответы:\n{joined}"
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    answer = response.content
    if _is_empty_partial(answer):
        if len(partials) != 1:
            return _refusal()
        answer = partials[0]

    sources: set[str] = set()
    if aggregate_intent and requested_statistics:
        for line in question_lines:
            sources.update(aggregate_sources.get(line, {}).get(aggregate_intent, set()))
    else:
        for community in useful_communities:
            nodes = set(community.nodes)
            for node in nodes:
                if node in kb.graph:
                    sources.update(_document_sources(kb.graph.nodes[node].get("sources", [])))
            for source, target, attrs in kb.graph.edges(data=True):
                if source in nodes and target in nodes:
                    sources.update(_document_sources(attrs.get("sources", [])))
    return Answer(answer, sorted(sources))
