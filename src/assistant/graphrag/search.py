"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено."
MAX_FACTS_LENGTH = 8000


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
    partials: list[str] = []
    useful_communities = []
    for community in kb.communities:
        prompt = (
            "Ответь на вопрос только по сводке сообщества. Верни JSON с полями answer и score, "
            "где score от 0 до 10 показывает полезность сводки.\n\n"
            f"Вопрос: {question}\nСводка: {community.summary}"
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
        if score > 0 and answer.strip():
            partials.append(answer)
            useful_communities.append(community)

    if not partials:
        return _refusal()

    joined = "\n".join(f"{number}. {answer}" for number, answer in enumerate(partials, start=1))
    prompt = (
        "Объедини частичные ответы в один краткий ответ на вопрос. Используй только приведённые данные.\n\n"
        f"Вопрос: {question}\n\nЧастичные ответы:\n{joined}"
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    if not response.content.strip():
        return _refusal()

    sources: set[str] = set()
    for community in useful_communities:
        nodes = set(community.nodes)
        for node in nodes:
            if node in kb.graph:
                sources.update(_document_sources(kb.graph.nodes[node].get("sources", [])))
        for source, target, attrs in kb.graph.edges(data=True):
            if source in nodes and target in nodes:
                sources.update(_document_sources(attrs.get("sources", [])))
    return Answer(response.content, sorted(sources))
