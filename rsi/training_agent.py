"""Agent-requested bounded training through the native trainer, without shell tools."""
from __future__ import annotations
import math
import json


def training_metrics(log):
    """Return only finite native trainer metrics, never filesystem/log text."""
    rows = []
    for line in log.splitlines():
        try:
            value = json.loads(line.strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or type(value.get('step')) is not int:
            continue
        row = {'step': value['step']}
        for key in ('loss', 'grad_norm', 'lr', 'action_flow/weighted_loss',
                    'video_flow/weighted_loss', 'hotload/version'):
            number = value.get(key)
            if type(number) in (float, int) and math.isfinite(number):
                row[key] = number
        if 'loss' in row:
            rows.append(row)
    return rows[-32:]


class TrainingBudget:
    def __init__(self, max_jobs=2, max_steps=2):
        if type(max_jobs) is not int or type(max_steps) is not int or min(max_jobs, max_steps) < 1:
            raise ValueError('Positive integer budgets are required')
        self.max_jobs, self.max_steps = max_jobs, max_steps
        self.requests = []

    def reserve(self, request):
        if set(request) != {'operation', 'learning_rate', 'steps', 'reason'}:
            raise ValueError('Only operation, learning_rate, steps and reason are allowed')
        operation, lr, steps = request['operation'], request['learning_rate'], request['steps']
        if operation != ('train' if not self.requests else 'adjust'):
            raise ValueError('Use train first, then adjust a subsequent job')
        if len(self.requests) >= self.max_jobs:
            raise ValueError('Training job budget exhausted')
        if type(lr) not in (float, int) or not math.isfinite(lr) or not 1e-7 <= lr <= 1e-4:
            raise ValueError('Learning rate must be finite and within [1e-7, 1e-4]')
        if type(steps) is not int or not 1 <= steps <= self.max_steps:
            raise ValueError('Training step budget exceeded')
        if not isinstance(request['reason'], str) or not 1 <= len(request['reason']) <= 2000:
            raise ValueError('A bounded explanation is required')
        if self.requests and lr == self.requests[-1]['learning_rate']:
            raise ValueError('An adjustment must change the learning rate')
        # Reserve before launching: failures still consume the attempt budget.
        self.requests.append(dict(request))
        return dict(request)


TRAINING_TOOL = {'type': 'function', 'function': {
    'name': 'propose_training',
    'description': 'Return a training-parameter proposal only; this inference bridge executes nothing. '
                   'The authorized caller validates this proposal and may run a bounded training job on its reserved GPU. '
                   'The first operation is train; adjust changes the learning rate for a later job. '
                   'No running job or deployed model is modified.',
    'parameters': {'type': 'object', 'additionalProperties': False,
        'properties': {'operation': {'type': 'string', 'enum': ['train', 'adjust']},
            'learning_rate': {'type': 'number', 'minimum': 1e-7, 'maximum': 1e-4},
            'steps': {'type': 'integer', 'minimum': 1, 'maximum': 2},
            'reason': {'type': 'string'}},
        'required': ['operation', 'learning_rate', 'steps', 'reason']}}}

FINISH_TOOL = {'type': 'function', 'function': {
    'name': 'finish_training_check',
    'description': 'Return a stop proposal and summarize supplied results only. The inference bridge executes nothing.',
    'parameters': {'type': 'object', 'additionalProperties': False,
        'properties': {'summary': {'type': 'string'}}, 'required': ['summary']}}}
