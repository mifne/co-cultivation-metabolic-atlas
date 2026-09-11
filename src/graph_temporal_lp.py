"""Experimental full-LP bipartite GNN + optional causal GRU proposals.

The graph is made from INPUT coefficients, not from a current FBA solution.
All LP rows/variables (including auxiliary constraints) are retained. There is
no PCA decoder, CPU optimizer, acceptance gate, or automatic PPO integration.
Predictions require an independent original-LP certificate/corrector.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gpu_pdhg_corrector import _validated_problem


def _signed_log(value):
    return value.sign() * torch.log1p(value.abs())


@dataclass(frozen=True)
class LPGraphBatch:
    identity: str
    problem_hashes: tuple[str, ...]
    row: torch.Tensor
    column: torch.Tensor
    coefficient: torch.Tensor
    rhs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    cost: torch.Tensor
    neq: int

    @classmethod
    def from_problems(cls, problems, *, stage, model_identity, device='cuda'):
        """Owned input snapshot with a union sparse pattern across environments.

        Stage and model identity are mandatory caller provenance, not inferred
        from numerical similarity. A different pattern requires new history.
        Host CSR validation/assembly is setup, not device-resident simulation.
        """
        if stage not in ('maxmin', 'aggregate', 'exchange') or not model_identity:
            raise ValueError('Explicit LP stage and immutable model identity required')
        problems = [_validated_problem(p) for p in problems]
        if not problems:
            raise ValueError('A nonempty LP batch is required')
        m, n = problems[0][0].shape
        neq = problems[0][-1]
        if any(p[0].shape != (m, n) or p[-1] != neq for p in problems):
            raise ValueError('Shared LP dimensions/equality count required')
        if any(np.isposinf(p[2]).any() or np.isneginf(p[3]).any() for p in problems):
            raise ValueError('Impossible infinite bounds')
        keys = [np.repeat(np.arange(m, dtype=np.int64), np.diff(p[0].indptr))*n
                + p[0].indices for p in problems]
        union = np.unique(np.concatenate(keys))
        coefficients = np.zeros((len(problems), len(union)))
        for i, (p, key) in enumerate(zip(problems, keys)):
            coefficients[i, np.searchsorted(union, key)] = p[0].data
        digest = hashlib.sha256(json.dumps(dict(stage=stage, model_identity=model_identity,
            rows=m, columns=n, equalities=neq, schema='input_bipartite_full_lp_v1'),
            sort_keys=True).encode())
        digest.update(union.astype('<i8').tobytes())
        def tensor(value, dtype=torch.float64):
            return torch.tensor(value, dtype=dtype, device=device)
        from .lp_trace import problem_hash
        return cls(digest.hexdigest(), tuple(problem_hash(p) for p in problems), tensor(union//n, torch.int64),
                   tensor(union % n, torch.int64), tensor(coefficients),
                   *(tensor(np.stack([p[k] for p in problems])) for k in (1, 2, 3, 4)), neq)

    def features(self, dtype):
        """Finite masks distinguish unbounded variables from a zero bound."""
        lower_finite, upper_finite = self.lower.isfinite(), self.upper.isfinite()
        lower = torch.where(lower_finite, self.lower, 0.)
        upper = torch.where(upper_finite, self.upper, 0.)
        variables = torch.stack((_signed_log(self.cost), _signed_log(lower),
            _signed_log(upper), lower_finite, upper_finite,
            lower_finite & upper_finite & (lower == upper)), dim=-1).to(dtype)
        equal = torch.arange(self.rhs.shape[1], device=self.rhs.device) < self.neq
        rows = torch.stack((_signed_log(self.rhs), equal.expand_as(self.rhs)), dim=-1).to(dtype)
        return variables, rows, _signed_log(self.coefficient).to(dtype)


@dataclass(frozen=True)
class GraphState:
    identity: str
    variables: torch.Tensor
    rows: torch.Tensor

    def detach(self):
        return GraphState(self.identity, self.variables.detach(), self.rows.detach())


@dataclass(frozen=True)
class GraphProposal:
    x: torch.Tensor
    y: torch.Tensor
    state: GraphState


def differentiable_lp_residuals(graph, x, y):
    """Full-space physics/optimality losses, NOT an acceptance certificate.

    Suitable for a training loss alongside trajectory-split teacher targets.
    The max/sum reductions intentionally retain rare large violations; average
    flux MSE alone is insufficient. Training may weight these terms, but cannot
    change the independent runtime certificate thresholds.
    """
    if (x.shape != graph.cost.shape or y.shape != graph.rhs.shape
            or x.device != graph.cost.device or y.device != graph.cost.device):
        raise ValueError('Proposal dimensions/device differ from the full LP')
    x, y = x.double(), y.double()
    activity = torch.zeros_like(graph.rhs)
    activity.index_add_(1, graph.row, graph.coefficient*x[:, graph.column])
    adjoint = torch.zeros_like(graph.cost)
    adjoint.index_add_(1, graph.column, graph.coefficient*y[:, graph.row])
    reduced = graph.cost-adjoint
    row_error = activity-graph.rhs
    eq = torch.arange(graph.rhs.shape[1], device=x.device) < graph.neq
    row_violation = torch.where(eq, row_error.abs(), row_error.clamp_min(0.))
    lower_finite, upper_finite = graph.lower.isfinite(), graph.upper.isfinite()
    lo, hi = torch.where(lower_finite, graph.lower, x), torch.where(upper_finite, graph.upper, x)
    bound_violation = torch.maximum((lo-x).clamp_min(0.), (x-hi).clamp_min(0.))
    target = torch.where(reduced >= 0., lo, hi)
    bound_gap = (reduced*(x-target)).abs().sum(1)
    row_gap = torch.where(eq, 0., y*row_error).abs().sum(1)
    variable_dual_violation = torch.maximum(
        torch.where(lower_finite, 0., reduced).clamp_min(0.),
        torch.where(upper_finite, 0., -reduced).clamp_min(0.))
    row_dual_violation = torch.where(eq, 0., y).clamp_min(0.)
    zero = torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)
    return dict(primal_residual=torch.cat((zero, row_violation, bound_violation), 1).amax(1),
        dual_violation=torch.cat((zero, variable_dual_violation, row_dual_violation), 1).amax(1),
        relative_kkt_gap=(bound_gap+row_gap)/(graph.cost*x).sum(1).abs().clamp_min(1.))


class GraphTemporalLP(nn.Module):
    """Signed message passing on LP edges followed by per-node temporal GRUs.

    Hidden width and message rounds are intentionally small: the objective is
    minimum time to a certified solve, not maximum NN parameter count. No graph
    convolution mixes environments. GNN-only is an explicit ablation.
    """
    def __init__(self, *, hidden=32, rounds=2, temporal=True):
        super().__init__()
        if (type(hidden) is not int or hidden < 1 or type(rounds) is not int
                or rounds < 1 or type(temporal) is not bool):
            raise ValueError('Positive integer hidden/rounds and bool temporal required')
        self.hidden, self.rounds, self.temporal = hidden, rounds, temporal
        self.variable_encoder = nn.Linear(6, hidden)
        self.row_encoder = nn.Linear(2, hidden)
        self.to_row = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(rounds))
        self.to_variable = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(rounds))
        self.row_updates = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(rounds))
        self.variable_updates = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(rounds))
        if temporal:
            self.variable_gru, self.row_gru = nn.GRUCell(hidden, hidden), nn.GRUCell(hidden, hidden)
        self.primal_head, self.dual_head = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, graph, previous=None):
        if graph.cost.device != self.primal_head.weight.device:
            raise ValueError('Graph and model must share a device')
        variables, rows, weights = graph.features(self.primal_head.weight.dtype)
        v, r = self.encode_nodes(graph,variables,rows)
        if previous is not None:
            if (not self.temporal or previous.identity != graph.identity
                    or previous.variables.shape != v.shape or previous.rows.shape != r.shape
                    or previous.variables.device != v.device or previous.rows.device != r.device
                    or previous.variables.dtype != v.dtype or previous.rows.dtype != r.dtype):
                raise ValueError('Temporal state identity/shape/device/dtype mismatch')
        # Degree normalization is per environment; zero coefficients contribute
        # no edge mass even if another environment needs that union-pattern edge.
        row_degree = weights.new_ones(weights.shape[0], r.shape[1])
        var_degree = weights.new_ones(weights.shape[0], v.shape[1])
        row_degree.index_add_(1, graph.row, weights.abs())
        var_degree.index_add_(1, graph.column, weights.abs())
        for j in range(self.rounds):
            row_message = torch.zeros_like(r)
            row_message.index_add_(1, graph.row,
                self.to_row[j](v)[:, graph.column]*weights.unsqueeze(-1))
            r = r + F.silu(self.row_updates[j](torch.cat((r, row_message/row_degree.unsqueeze(-1)), -1)))
            var_message = torch.zeros_like(v)
            var_message.index_add_(1, graph.column,
                self.to_variable[j](r)[:, graph.row]*weights.unsqueeze(-1))
            v = v + F.silu(self.variable_updates[j](torch.cat((v, var_message/var_degree.unsqueeze(-1)), -1)))
        if self.temporal:
            old_v = torch.zeros_like(v) if previous is None else previous.variables
            old_r = torch.zeros_like(r) if previous is None else previous.rows
            v = self.variable_gru(v.flatten(0, 1), old_v.flatten(0, 1)).reshape_as(v)
            r = self.row_gru(r.flatten(0, 1), old_r.flatten(0, 1)).reshape_as(r)
        raw_x, raw_y = self.primal_head(v).squeeze(-1).double(), self.dual_head(r).squeeze(-1).double()
        x,y=self.decode_heads(graph,raw_x,raw_y)
        return GraphProposal(x, y, GraphState(graph.identity, v, r))

    def encode_nodes(self,graph,variables,rows):
        return F.silu(self.variable_encoder(variables)), F.silu(self.row_encoder(rows))

    def decode_heads(self,graph,raw_x,raw_y):
        lo_ok, hi_ok = graph.lower.isfinite(), graph.upper.isfinite()
        lo, hi = torch.where(lo_ok, graph.lower, 0.), torch.where(hi_ok, graph.upper, 0.)
        # This enforces the box ONLY, not stoichiometric feasibility/optimality.
        x = torch.where(lo_ok & hi_ok, lo+(hi-lo)*raw_x.sigmoid(),
            torch.where(lo_ok, lo+F.softplus(raw_x), torch.where(hi_ok, hi-F.softplus(raw_x), raw_x)))
        is_eq = torch.arange(raw_y.shape[1], device=raw_y.device) < graph.neq
        y = torch.where(is_eq, raw_y, -F.softplus(raw_y))
        return x,y


class GraphTemporalSession:
    """Transactional inference history: only certified outcomes may commit.

    Callers supply the unchanged original-LP certificate result. This class is
    not a certificate implementation. Session tokens reject stale commits after
    reset or another commit. Training uses the differentiable model directly.
    """
    def __init__(self, model):
        if not model.temporal:
            raise ValueError('Use GNN-only directly without a temporal session')
        self.model = model
        self._history = {}
        self._generation = 0

    @staticmethod
    def _ids(values):
        ids = tuple(values)
        if (not ids or len(set(ids)) != len(ids)
                or any(type(i) not in (int, str) for i in ids)):
            raise ValueError('Nonempty unique integer/string environment IDs required')
        return ids

    def reset(self, environment_ids=None):
        ids = None if environment_ids is None else self._ids(environment_ids)
        if ids is None:
            self._history.clear()
        else:
            for key in ids:
                self._history.pop(key, None)
        self._generation += 1

    @torch.inference_mode()
    def propose(self, graph, environment_ids):
        ids = self._ids(environment_ids)
        if len(ids) != graph.cost.shape[0]:
            raise ValueError('One ID per environment required')
        batch, n = graph.cost.shape
        m = graph.rhs.shape[1]
        dtype = self.model.primal_head.weight.dtype
        v = torch.zeros((batch, n, self.model.hidden), device=graph.cost.device, dtype=dtype)
        r = torch.zeros((batch, m, self.model.hidden), device=graph.cost.device, dtype=dtype)
        for i, key in enumerate(ids):
            if key in self._history:
                old = self._history[key]
                if old.identity != graph.identity:
                    raise ValueError('Reset history before changing LP graph/model/stage identity')
                v[i], r[i] = old.variables, old.rows
        proposal = self.model(graph, GraphState(graph.identity, v, r))
        return proposal, (self._generation, ids, graph.identity, id(self), id(proposal))

    def commit(self, token, proposal, certified):
        generation, ids, identity, session_id, proposal_id = token
        mask = np.asarray(certified)
        if (generation != self._generation or identity != proposal.state.identity
                or session_id != id(self) or proposal_id != id(proposal)
                or mask.dtype != np.bool_ or mask.shape != (len(ids),)
                or proposal.x.shape[0] != len(ids)):
            raise ValueError('Stale/mismatched commit or invalid original-certificate mask')
        # No partially-mutated history if any accepted state is corrupt.
        for i, ok in enumerate(mask):
            if ok and not all(bool(t[i].isfinite().all()) for t in
                (proposal.x, proposal.y, proposal.state.variables, proposal.state.rows)):
                raise ValueError('Cannot commit nonfinite predictions')
        for i, (key, ok) in enumerate(zip(ids, mask)):
            if ok:
                self._history[key] = GraphState(identity, proposal.state.variables[i].detach().clone(),
                                                proposal.state.rows[i].detach().clone())
            else:
                # A failed step creates a gap in the accepted trajectory.
                self._history.pop(key, None)
        self._generation += 1
