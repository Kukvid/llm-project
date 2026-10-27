"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if not graph:
        return []

    partition = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=7)
    groups = sorted((sorted(nodes) for nodes in partition), key=lambda nodes: nodes[0])
    return [Community(community_id, nodes) for community_id, nodes in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    nodes = set(community.nodes)
    entity_counts = Counter(graph.nodes[node]["type"] for node in community.nodes if node in graph)
    edges = sorted(
        (
            graph.nodes[source]["name"],
            attrs["type"],
            graph.nodes[target]["name"],
            attrs.get("weight", 1),
        )
        for source, target, attrs in graph.edges(data=True)
        if source in nodes and target in nodes
    )
    counts = ", ".join(f"{kind}: {count}" for kind, count in sorted(entity_counts.items())) or "нет"
    facts = "\n".join(f"- {source} -{kind}-> {target}; упоминаний: {weight}" for source, kind, target, weight in edges)
    prompt = (
        "Напиши по-русски связную сводку сообщества графа знаний длиной не более 150 слов. "
        "Не придумывай факты и числа: используй только готовые данные ниже.\n\n"
        f"Сущностей: {len(community.nodes)}. Связей: {len(edges)}. Типы сущностей: {counts}.\n"
        f"Факты:\n{facts or '- связей нет'}"
    )
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=512)
    return " ".join(response.content.split()[:150])
