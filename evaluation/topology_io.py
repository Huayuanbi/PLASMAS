"""Shared topology parsing and graph validity checks; no pilot dependencies."""
import json
import networkx as nx

def role_text(nodes: list[dict]) -> str:
    return "\n\n".join(
        f"ID: {node['id']}\nROLE: {node['role']}\nROLE BRIEF:\n{node['role_brief']}"
        for node in nodes
    )

def user_prompt(task: str, nodes: list[dict]) -> str:
    return (
        "MATHEMATICS PROBLEM:\n"
        + task
        + "\n\nAVAILABLE AGENTS:\n"
        + role_text(nodes)
        + "\n\nSelect the topology now."
    )

def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")

def validate_topology(
    payload: dict, nodes: list[dict], *, require_difficulty: bool = True
) -> dict:
    valid_ids = [str(node["id"]) for node in nodes]
    selected = payload.get("selected_agents")
    edges = payload.get("edges")
    difficulty = payload.get("difficulty_estimate")
    if require_difficulty and difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty_estimate must be easy, medium, or hard")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise ValueError("selected_agents must be a list of agent IDs")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_agents contains duplicates")
    if not selected or any(item not in valid_ids for item in selected):
        raise ValueError("selected_agents contains an unknown agent ID")
    if "agent_5" not in selected:
        raise ValueError("agent_5 must be selected")
    if not isinstance(edges, list):
        raise ValueError("edges must be a list")
    normalized_edges: list[tuple[str, str]] = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(item, str) for item in edge)
        ):
            raise ValueError("each edge must be a two-element agent-ID list")
        source, target = edge
        if source not in selected or target not in selected:
            raise ValueError("every edge endpoint must be selected")
        if source == target:
            raise ValueError("self-edges are not allowed")
        normalized_edges.append((source, target))
    if len(normalized_edges) != len(set(normalized_edges)):
        raise ValueError("edges contains duplicates")

    graph = nx.DiGraph()
    graph.add_nodes_from(selected)
    graph.add_edges_from(normalized_edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("edges must form a DAG")
    for node_id in selected:
        if node_id != "agent_5" and not nx.has_path(graph, node_id, "agent_5"):
            raise ValueError(f"{node_id} has no directed path to agent_5")
    result = {
        "selected_agents": [node_id for node_id in valid_ids if node_id in selected],
        "edges": [list(edge) for edge in normalized_edges],
    }
    if require_difficulty:
        result["difficulty_estimate"] = difficulty
    return result
