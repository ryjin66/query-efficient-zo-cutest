"""Recording adapters for the retained experiment protocols."""
import json
import math
from pathlib import Path

from .logging_utils import RunRecorder
from .sampled_logging import RunRecorder as SampledRecorder


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


class SampledRunRecorder(SampledRecorder):
    def mark_status(self, status):
        # Query-limited sampled traces do not append an unsampled final point.
        if status != 'completed' and self.rows:
            self.rows[-1]['status'] = status


class BudgetRecorder(RunRecorder):
    """Retain vectors belonging to the last in-budget recorded snapshot."""

    def __init__(self, *args, budget, time_cap, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget = budget
        self.time_cap = time_cap
        self.scored = None

    def capture(self):
        if not self.rows:
            return
        row = self.rows[-1]
        if row['query'] > self.budget or (self.time_cap and row['elapsed_seconds'] > self.time_cap):
            return
        if self._last_snapshot['query'] != row['query']:
            return
        self.scored = dict(row)
        self.scored.update(best_x=None if self.best_x is None else self.best_x.tolist(),
                           best_violation_x=None if self.best_violation_x is None else self.best_violation_x.tolist(),
                           query_budget=self.budget, max_time_seconds=self.time_cap)

    def record(self, *args, **kwargs):
        super().record(*args, **kwargs)
        self.capture()

    def mark_status(self, status):
        super().mark_status(status)
        self.capture()

    def save_incumbent(self, path):
        super().save_incumbent(path)
        if self.scored is not None:
            path = Path(path)
            write_json(path.with_name(path.stem+'.in_budget.json'), self.scored)


def captured_recorder_class(base, instances):
    """Preserve the last finite, in-budget vector when a run terminates."""
    class CapturedRecorder(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.finite_scored = None
            instances.append(self)

        def capture(self):
            super().capture()
            if self.scored is not None and math.isfinite(self.scored['best_feasible_f']):
                self.finite_scored = dict(self.scored)

        def to_csv(self, path):
            path = Path(path)
            temporary = path.with_name(path.name+'.tmp')
            super().to_csv(temporary)
            temporary.replace(path)

        def save_incumbent(self, path):
            super().save_incumbent(path)
            if self.finite_scored is not None:
                path = Path(path)
                write_json(path.with_name(path.stem+'.finite_in_budget.json'), self.finite_scored)
    return CapturedRecorder
