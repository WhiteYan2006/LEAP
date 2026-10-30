"""Engine-independent selection of already published scores; never reads truth."""
from dataclasses import dataclass
import math
from .ledger import Ledger


@dataclass(frozen=True)
class RequestState:
    request_id: str
    slot: int
    generation: int
    generated: int
    budget: int
    arrival_ns: int = 0


def select_published(states, ledger: Ledger, *, now_ns, decode_capacity, mode='countdown'):
    """Stable lower-score-first decode selection with whole-snapshot FIFO fallback.

    KV allocation, prefill, and admission stay with the serving engine.
    """
    if mode not in ('countdown', 'static_total', 'fcfs') or type(decode_capacity) is not int or decode_capacity < 1:
        raise ValueError('Invalid selection mode or decode capacity')
    states = list(states)
    if len({s.request_id for s in states}) != len(states): raise ValueError('Duplicate live request IDs')
    rows = []
    for state in states:
        if not 0 <= state.generated < state.budget: raise ValueError('Only live requests are eligible')
        snap = ledger.snapshot(state, 'ttt_rls', now_ns) if mode != 'fcfs' else dict(score=None, source_value=None)
        score = snap['source_value'] if mode == 'static_total' else snap['score']
        if score is not None and not math.isfinite(score): raise ValueError('Nonfinite published score')
        rows.append((state, score, snap))
    fifo = mode == 'fcfs' or any(score is None for _, score, _ in rows)
    ordered = sorted(rows, key=lambda row: ((row[0].arrival_ns if fifo else row[1]), row[0].arrival_ns, row[0].request_id))
    return [row[0] for row in ordered[:decode_capacity]], dict(fifo_fallback=fifo, snapshot_ns=now_ns,
                                                             published_scores={s.request_id: score for s, score, _ in rows})

