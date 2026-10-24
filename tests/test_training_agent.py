import unittest
from rsi.training_agent import TrainingBudget, training_metrics


class TrainingBudgetTests(unittest.TestCase):
    def request(self, **kw):
        return dict(operation='train', learning_rate=1e-5, steps=2, reason='bounded smoke test', **kw)

    def test_job_limit_and_real_adjustment(self):
        b = TrainingBudget()
        b.reserve(self.request())
        with self.assertRaises(ValueError):
            b.reserve(dict(operation='adjust', learning_rate=1e-5, steps=2, reason='unchanged'))
        b.reserve(dict(operation='adjust', learning_rate=5e-6, steps=1, reason='lower rate'))
        with self.assertRaises(ValueError):
            b.reserve(dict(operation='adjust', learning_rate=2e-6, steps=1, reason='extra job'))

    def test_rejects_injection_nonfinite_and_bool(self):
        for change in ({'steps': True}, {'steps': 3}, {'learning_rate': float('nan')},
                       {'learning_rate': True}, {'learning_rate': 1.}, {'path': '/root'},
                       {'operation': 'adjust'}, {'reason': ''}):
            r = self.request(); r.update(change)
            b = TrainingBudget()
            with self.subTest(change=change), self.assertRaises(ValueError): b.reserve(r)
            self.assertEqual(b.requests, [])

    def test_metrics_are_finite_allowlisted_numbers(self):
        rows = training_metrics('progress 1/2\n'
            '{"step": 1, "loss": 0.2, "lr": 0.000001, "secret_path": "/private"}\n'
            '{"step": 2, "loss": NaN, "grad_norm": 0.3}\n'
            '{"step": true, "loss": 0.4}\n'
            '{"step": 3, "loss": 0.1, "grad_norm": Infinity}')
        self.assertEqual(rows, [{'step': 1, 'loss': 0.2, 'lr': 0.000001},
                                {'step': 3, 'loss': 0.1}])
