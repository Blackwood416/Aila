import tempfile
import unittest
from pathlib import Path
from q4e_hotset_profile import read, merge

class Profiles(unittest.TestCase):
    def test_merge_reorders_existing_entries(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,out = [Path(d)/n for n in ('a','b','out')]
            a.write_text('AILA_Q4E_HOTSET 1\nid\n1 2 2\n0 2 0 1\n1 0 3 1\n')
            b.write_text('AILA_Q4E_HOTSET 1\nid\n1 2 2\n1 5 0 1\n0 0 0 0\n')
            merge([a,b],out)
            self.assertEqual(read(out)['rows'],[(1,5,3,2),(0,2,0,1)])
            b.write_text(b.read_text().replace('\nid\n','\nwrong\n'))
            with self.assertRaises(ValueError): merge([a,b],out)
    def test_duplicate_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'p'
            for key in (0,2,-1):
                p.write_text(f'AILA_Q4E_HOTSET 1\nid\n1 2 2\n0 1 0 1\n{key} 0 0 0\n')
                with self.assertRaises(ValueError): read(p)

if __name__ == '__main__': unittest.main()
