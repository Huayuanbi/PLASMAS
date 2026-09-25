"""Differentiable auxiliary structure supervision for JSON topology DPO."""
import json
import torch
from torch import nn
from torch.nn import functional as F


def graph_targets(text, node_ids):
    obj = json.loads(text)
    selected, edges = obj['selected_agents'], obj['edges']
    ids = {v: i for i, v in enumerate(node_ids)}
    if len(set(selected)) != len(selected) or not set(selected) <= ids.keys():
        raise ValueError('invalid selected agents')
    if node_ids[-1] not in selected:
        raise ValueError('finalizer must be active')
    edge_set = {tuple(e) for e in edges}
    if len(edge_set) != len(edges):
        raise ValueError('duplicate edges')
    nodes = torch.tensor([float(v in selected) for v in node_ids])
    adjacency = torch.zeros(len(ids), len(ids))
    successors = {v: [] for v in selected}
    for a, b in edge_set:
        if a not in selected or b not in selected or a == b or a == node_ids[-1]:
            raise ValueError('invalid edge')
        successors[a].append(b)
        adjacency[ids[a], ids[b]] = 1
    visited, visiting = set(), set()
    def visit(v):
        if v in visiting:
            raise ValueError('cyclic graph')
        if v in visited:
            return
        visiting.add(v)
        if not successors[v] and v != node_ids[-1]:
            raise ValueError('node cannot reach finalizer')
        for w in successors[v]:
            visit(w)
        visiting.remove(v)
        visited.add(v)
    for v in selected:
        visit(v)
    return nodes, adjacency


class GraphHead(nn.Module):
    def __init__(self, hidden_size, num_nodes=6):
        super().__init__()
        self.num_nodes = num_nodes
        self.projection = nn.Linear(hidden_size, num_nodes + num_nodes**2)

    def forward(self, hidden):
        values = self.projection(hidden.float())
        return values[:, :self.num_nodes], values[:, self.num_nodes:].reshape(-1, self.num_nodes, self.num_nodes)


def graph_scores(node_logits, edge_logits, targets, node_weight=1., edge_weight=1.):
    nodes, edges = targets
    n = node_logits.shape[-1]
    domain = ~torch.eye(n, dtype=torch.bool, device=edge_logits.device)
    domain[-1] = False  # finalizer has no outgoing edges; self-edges forbidden
    node_loss = F.binary_cross_entropy_with_logits(node_logits[:, :-1], nodes[:, :-1], reduction='none').mean(-1)
    edge_loss = F.binary_cross_entropy_with_logits(edge_logits[:, domain], edges[:, domain], reduction='none').mean(-1)
    return -node_weight * node_loss - edge_weight * edge_loss


def graph_pair_loss(chosen_score, rejected_score, margin=0., temperature=1.):
    if temperature <= 0 or margin < 0:
        raise ValueError('invalid graph margin or temperature')
    return F.softplus((margin + rejected_score - chosen_score) / temperature)


def prompt_batch(batch):
    """Slice chosen rows to prompt only, right-padding shorter prompts."""
    count = batch['weights'].numel()
    labels = batch['labels'][:count]
    lengths = (labels != -100).long().argmax(-1)
    if (lengths <= 0).any():
        raise ValueError('empty prompt or missing completion')
    width = int(lengths.max())
    ids = batch['input_ids'][:count, :width].clone()
    mask = torch.arange(width, device=ids.device)[None, :] < lengths[:, None]
    ids.masked_fill_(~mask, 0)
    return ids, mask.long(), lengths


def chosen_token_nll(chosen_logps, labels):
    counts = (labels[:, 1:] != -100).sum(-1).clamp_min(1)
    return -chosen_logps / counts
