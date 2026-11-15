from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from continual_ai.agent import learning_round
from continual_ai.storage import CloudStore, LocalStore
from continual_ai.tasks import task_for_seed, parse_integer_answer, verify


class FakeClient:
    model = "fake-test-model"
    def generate(self, prompt, **kwargs):
        # This fake model only exists to test storage and round orchestration.
        for seed in range(100):
            task = task_for_seed(seed)
            if task.question in prompt.split("Question: ")[-1]:
                return f"ANSWER: {task.expected}"
        raise ValueError("Unexpected task")


class TaskTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        for seed in range(36):
            self.assertEqual(task_for_seed(seed), task_for_seed(seed))
    def test_answer_parser_rejects_ambiguity(self):
        self.assertEqual(parse_integer_answer("ANSWER: -22"), -22)
        self.assertIsNone(parse_integer_answer("ANSWER: 1\nANSWER: 2"))
        self.assertIsNone(parse_integer_answer("The answer is 5"))
        self.assertIsNone(parse_integer_answer("ANSWER: 4.3"))
    def test_verify(self):
        task = task_for_seed(5)
        self.assertTrue(verify(task, f"ANSWER: {task.expected}"))
        self.assertFalse(verify(task, f"ANSWER: {task.expected + 1}"))
    def test_no_holdout_seed_in_training(self):
        self.assertNotEqual(task_for_seed(0).seed, task_for_seed(9000000).seed)


class StoreTests(unittest.TestCase):
    def test_local_round_and_export(self):
        with tempfile.TemporaryDirectory() as root:
            store = LocalStore(Path(root)/"db.sqlite")
            self.assertEqual(store.get_next_seed(), 0)
            self.assertTrue(learning_round(store, FakeClient())["correct"])
            self.assertEqual(store.get_next_seed(), 1)
            self.assertEqual(store.stats()["total"], 1)
            rows = store.training_examples()
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["verified"])
            store.save_chat("user", "Hello")
            store.save_chat("assistant", "Welcome")
            self.assertEqual(store.last_chat(1), [("Assistant", "Welcome")])
            store.close()
            again = LocalStore(Path(root)/"db.sqlite")
            self.assertEqual(again.get_next_seed(), 1)
            again.close()

    def test_cloud_round_resume(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            cloud = CloudStore(folder)
            self.assertTrue(learning_round(cloud, FakeClient())["correct"])
            self.assertEqual(cloud.get_next_seed(), 1)
            again = CloudStore(folder)
            self.assertEqual(again.get_next_seed(), 1)
            self.assertTrue(learning_round(again, FakeClient())["correct"])
            again.write_summary()
            self.assertEqual(again.stats()["total"], 2)
            self.assertEqual(len((folder/"verified_examples.jsonl").read_text().splitlines()), 2)
            self.assertIn("Neural weights have NOT", (folder/"summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
