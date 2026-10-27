"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    node_sources: dict[str, set[str]] = {}
    edge_sources: dict[tuple[str, str], set[str]] = {}
    edge_types: dict[tuple[str, str], str] = {}

    for result in results:
        for entity in result.entities:
            graph.add_node(entity.node_id, type=entity.type, name=entity.name)
            node_sources.setdefault(entity.node_id, set()).add(result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                graph.add_node(entity.node_id, type=entity.type, name=entity.name)
                node_sources.setdefault(entity.node_id, set()).add(result.chunk_id)
            edge = (relation.subject.node_id, relation.object.node_id)
            edge_types[edge] = relation.type
            edge_sources.setdefault(edge, set()).add(result.chunk_id)

    for node, sources in node_sources.items():
        graph.nodes[node]["sources"] = sorted(sources)
    for (source, target), sources in edge_sources.items():
        graph.add_edge(source, target, type=edge_types[(source, target)], weight=len(sources), sources=sorted(sources))

    conflicts = 0
    seen: set[frozenset[str]] = set()
    for source, target, attrs in list(graph.edges(data=True)):
        pair = frozenset((source, target))
        if source == target or attrs["type"] != "REPLACED_BY" or pair in seen or not graph.has_edge(target, source):
            continue
        reverse = graph.edges[target, source]
        if reverse["type"] != "REPLACED_BY":
            continue
        seen.add(pair)
        conflicts += 1
        if attrs["weight"] > reverse["weight"]:
            graph.remove_edge(target, source)
        elif attrs["weight"] < reverse["weight"]:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

    graph.graph["conflicts"] = conflicts
    return graph
