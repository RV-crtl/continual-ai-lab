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
        with self.assertRaisesRegex(RuntimeError, 'empty FINAL answer twice'):
            c.generate('2+3?',max_tokens=192)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1]['options']['num_predict'],384)
    def test_second_chat_answer_accepted(self):
        c=OllamaClient('test')
        i=iter([{'message':{'content':''}}, {'message':{'content':'42'}}])
        c._json=lambda *args: next(i)
        self.assertEqual(c.generate('answer?'),'42')
    def test_empty_metric(self):
        with tempfile.TemporaryDirectory() as td:
            s=LocalStore(Path(td)/'l.db')
            t=task_for_seed(0)
            s.add(record(t,'','test',False))
            self.assertEqual(s.stats()['empty_responses'],1)
            s.close()

if __name__=='__main__':
    unittest.main()
