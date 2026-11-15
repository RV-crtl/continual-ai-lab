"""Regression tests: Qwen native chat, empty reply safeguards, parsing telemetry."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from continual_ai.ollama_client import OllamaClient
from continual_ai.storage import LocalStore
from continual_ai.tasks import task_for_seed
from continual_ai.storage import record

class InferenceTests(unittest.TestCase):
    def test_chat_top_level_no_thinking(self):
        c = OllamaClient('qwen3:4b')
        requests=[]
        def fake(path,payload):
            requests.append((path,payload))
            return {'message':{'content':'ANSWER: 5'},'done_reason':'stop','eval_count':6}
        c._json=fake
        self.assertEqual(c.generate('2+3?',max_tokens=192),'ANSWER: 5')
        self.assertEqual(requests[0][0], '/api/chat')
        self.assertIs(requests[0][1]['think'], False)
        self.assertEqual(requests[0][1]['options']['num_predict'],192)
    def test_empty_final_retried_once_then_stops(self):
        c=OllamaClient('qwen3:4b')
        calls=[]
        def empty(path,payload):
            calls.append(payload)
            return {'message':{'content':'','thinking':'private'},'done_reason':'length','eval_count':payload['options']['num_predict']}
        c._json=empty
        with self.assertRaisesRegex(RuntimeError, 'after two attempts'):
            c.generate('2+3?',max_tokens=192)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1]['options']['num_predict'],512)
    def test_second_chat_answer_accepted(self):
        c=OllamaClient('test')
        i=iter([{'message':{'content':''}}, {'message':{'content':'42'}}])
        c._json=lambda *args: next(i)
        self.assertEqual(c.generate('answer?'),'42')

    def test_math_answer_uses_schema_and_no_think(self):
        c = OllamaClient('qwen3:8b')
        requests=[]
        def fake(path,payload):
            requests.append(payload)
            return {'message':{'content':'{"answer": 12}'}, 'done_reason':'stop','eval_count':10}
        c._json=fake
        self.assertEqual(c.generate('324 divided by 27', answer_mode=True),'{"answer": 12}')
        self.assertIs(requests[0]['think'],False)
        self.assertEqual(requests[0]['format']['required'], ['answer'])
        self.assertTrue(requests[0]['messages'][-1]['content'].endswith('/no_think'))

    def test_truncated_looks_valid_but_is_not_recorded(self):
        c = OllamaClient('qwen3:4b')
        responses=iter([
            {'message':{'content':'{"answer": 12}'}, 'done_reason':'length', 'eval_count':256},
            {'message':{'content':'{"answer": 11}'}, 'done_reason':'stop', 'eval_count':12},
        ])
        c._json=lambda *args: next(responses)
        self.assertEqual(c.generate('task', answer_mode=True), '{"answer": 11}')
        self.assertEqual(c.last_metadata['attempt'],2)

    def test_reasoning_prose_without_final_answer_is_rejected(self):
        c=OllamaClient('qwen3:4b')
        c._json=lambda *args: {'message':{'content': 'Okay, let me think through this problem...'},
                               'done_reason':'stop','eval_count':100}
        with self.assertRaisesRegex(RuntimeError, 'complete, parseable'):
            c.generate('task', answer_mode=True)

    def test_empty_metric(self):
        with tempfile.TemporaryDirectory() as td:
            s=LocalStore(Path(td)/'l.db')
            t=task_for_seed(0)
            s.add(record(t,'','test',False))
            self.assertEqual(s.stats()['empty_responses'],1)
            s.close()

if __name__=='__main__':
    unittest.main()
