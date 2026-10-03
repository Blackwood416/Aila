import json
import tempfile
import unittest
from pathlib import Path
from q4e_prompt_hotset_audit import audit


class PromptAuditTests(unittest.TestCase):
    def run_audit(self, split):
        with tempfile.TemporaryDirectory() as d:
            profile=Path(d)/'profile'
            profile.write_text('AILA_Q4E_HOTSET 1\ntest\n1 4 4\n'+''.join(f'{i} 0 0 0\n' for i in range(4)))
            rows=[]
            for a,b in ((0,2),(2,2)):
                rows.append(dict(type='forward_begin',position=0))
                for count in ([2,2] if split else [4]):
                    rows.append(dict(type='route',layer=0,rows=count,top_k=2,ids=[a,a+1]*count))
                rows.append(dict(type='route',layer=0,rows=1,top_k=2,ids=[b,b+1]))
            trace=Path(d)/'trace'
            trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return audit(str(trace),str(profile),2)['requests']

    def test_no_future_or_next_prompt_leakage(self):
        rows=self.run_audit(False)
        self.assertEqual(rows[0]['candidates']['128']['coverage']['8'],0)
        self.assertEqual(rows[1]['candidates']['128']['coverage']['8'],1)

    def test_chunk_invariance(self):
        self.assertEqual(self.run_audit(False),self.run_audit(True))


if __name__=='__main__':
    unittest.main()
