"""No GPU, online connection, or model downloads required for unit tests."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from continual_ai.multi_agent import CallBudget, MultiAgentController, choose_strategy, VERIFIER_SEED_START, strategy_prompt
from continual_ai.storage import CloudStore, LocalStore
from continual_ai.tasks import task_for_seed
from continual_ai.train_lora import get_dataset
from continual_ai.training_registry import evaluate_gate


class FakeAI:
    model = 'test-model'
    def generate(self, prompt, **kwargs):
        # Identify the final task; any demonstration text must not affect it.
        for seed in list(range(20)) + [VERIFIER_SEED_START+i for i in range(40)]:
            task = task_for_seed(seed)
            if prompt.rstrip().endswith(task.question):
                return f'ANSWER: {task.expected}'
        raise AssertionError(f'Unknown problem prompt: {prompt[-150:]}')


class MultiAgentTests(unittest.TestCase):
    def test_prompt_strategies_avoid_placeholders(self):
        for strategy in ('few_shot', 'zero_shot', 'explicit_format'):
            prompt = strategy_prompt(task_for_seed(0), [], strategy)
            self.assertNotIn('<computed integer>', prompt)
            self.assertNotIn('<number>', prompt)

    def test_budget_protects_verification(self):
        b = CallBudget()
        self.assertEqual(b.verification_share, .5)
        with self.assertRaises(ValueError):
            CallBudget(learner=2, optimiser=2, verifier=1)

    def test_cloud_cycle_keeps_verifier_out_of_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            db=CloudStore(root)
            c=MultiAgentController(db, FakeAI(), root)
            result=c.cycle()
            self.assertGreaterEqual(result['verifier_total'], 2)
            self.assertLessEqual(result['verifier_total'], 10)
            self.assertEqual(result['verifier_correct'], result['verifier_total'])
            self.assertEqual(db.get_next_seed(), 2)
            self.assertEqual(db.stats()['total'], 2)
            self.assertEqual(len((root/'verified_examples.jsonl').read_text().splitlines()), 2)
            self.assertEqual(c.state['calls']['learner'], 1)
            self.assertEqual(c.state['calls']['optimiser'], 1)
            self.assertGreaterEqual(c.state['calls']['verifier'], 2)
            self.assertEqual(choose_strategy(c.state['strategies']), 'zero_shot')
            # Resume uses reserved verifier seeds from next cycle, never trains on them.
            c2=MultiAgentController(CloudStore(root), FakeAI(), root)
            c2.cycle()
            self.assertGreaterEqual(c2.state['calls']['verifier'], 4)
            self.assertEqual(CloudStore(root).stats()['total'], 4)
            self.assertEqual(len((root/'verified_examples.jsonl').read_text().splitlines()), 4)

    def test_local_db_and_verifier_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            s=LocalStore(root/'local.sqlite3')
            MultiAgentController(s, FakeAI(), root).cycle()
            self.assertEqual(len(s.training_examples()), 2)
            self.assertGreaterEqual(s.db.execute("SELECT COUNT(*) FROM attempts WHERE split='verify'").fetchone()[0],2)
            s.close()

    def test_training_dataset_revalidates_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'samples.jsonl'
            from continual_ai.storage import append_jsonl
            t=task_for_seed(1)
            base={'seed':t.seed,'kind':t.kind,'question':t.question,'expected':t.expected,
                  'label_source':'deterministic_generator','verified':True}
            append_jsonl(file, base)
            append_jsonl(file, {**base,'expected':t.expected+1,'seed':2})
            append_jsonl(file, {**base,'seed':9_000_000})
            self.assertEqual(len(get_dataset(file, 10)),1)

    def test_gate_rejects_incomplete_and_fake_audit(self):
        self.assertFalse(evaluate_gate({})[0])
        report={}
        for name,start in [('primary',9_000_000),('audit',9_100_000)]:
            samples=[{'seed':start+i,'kind':task_for_seed(start+i).kind,'correct':False} for i in range(60)]
            kinds={task_for_seed(start+i).kind:{'passed':0,'total':10} for i in range(6)}
            data={'n':60,'correct':0,'kinds':kinds,'samples':samples}
            report[name]={'baseline':dict(data), 'candidate':dict(data)}
        allowed, reasons=evaluate_gate(report)
        self.assertFalse(allowed)
        self.assertTrue(any('primary:' in reason for reason in reasons))
        report['primary']['candidate']['samples'][0]['seed']=0
        self.assertFalse(evaluate_gate(report)[0])


if __name__ == '__main__':
    unittest.main()
