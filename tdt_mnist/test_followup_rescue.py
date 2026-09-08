"""Behavioral check for the preregistered, one-retry rescue boundary."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from run_residual_followups import bp_run


class RescueTests(unittest.TestCase):
    def test_retry_only_failed_e20c_and_never_more_than_once(self):
        for condition,first_status,expected_calls in [('E20c','failed',2),('E20c','success',1),('E20a','failed',1),('E20b','failed',1)]:
            with self.subTest(condition=condition,status=first_status),tempfile.TemporaryDirectory() as directory:
                calls=[]
                def fake_attempt(c,seed,out,data,attempt):
                    calls.append((c,seed,attempt))
                    return dict(condition=c,seed=seed,status=first_status,attempt=attempt,test=None,elapsed_seconds=0.)
                with patch('run_residual_followups.bp_attempt',side_effect=fake_attempt):
                    bp_run(condition,2,Path(directory),Path('data'))
                self.assertEqual(len(calls),expected_calls)
                self.assertEqual(calls,[(condition,2,i) for i in range(expected_calls)])


if __name__=='__main__':unittest.main()
