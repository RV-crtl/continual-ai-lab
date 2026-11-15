from pathlib import Path
import tempfile
import unittest
from continual_ai.storage import LocalStore, record
from continual_ai.tasks import extract_integer_answer, task_for_seed

class RescoringTests(unittest.TestCase):
    def test_real_cloud_examples_rescored(self):
        for response, expected in [('343',343),('-342',-342),('-42',-42),
                                   ('12² - 4² = 144 - 16 = 128',128),
                                   ('ANSWER: 13',13),('ANSWER: 7',7),('x = -4',-4)]:
            self.assertEqual(extract_integer_answer(response)[0],expected)
        for invalid in ['ANSWER: <integer>', 'ANSWER: 1\nANSWER: 2',
                        'ANSWER: 4710%52=46', '-26.675675675675677']:
            self.assertIsNone(extract_integer_answer(invalid)[0])

    def test_rescore_legacy_failures_and_rolling_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            store=LocalStore(Path(d)/'history.sqlite3')
            a=task_for_seed(0)
            b=task_for_seed(1)
            c=task_for_seed(2)
            store.add(record(a, str(a.expected), 'test', False))
            store.add(record(b, 'ANSWER: '+str(b.expected), 'test', True))
            store.add(record(c, 'ANSWER: <integer>', 'test', False))
            s=store.stats()
            self.assertEqual(s['total'],3)
            self.assertEqual(s['legacy_correct'],1)
            self.assertEqual(s['correct'],2)
            self.assertEqual(s['rescued_from_format'],1)
            self.assertEqual(s['unparseable'],1)
            self.assertEqual(len(store.recent_attempts()),3)
            self.assertEqual(len(store.training_examples()),3)
            store.close()

if __name__=='__main__':
    unittest.main()
