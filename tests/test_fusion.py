import unittest
from benchmarks.fusion import fuse


class FusionTests(unittest.TestCase):
    def test_duplicate_ids_do_not_vote_twice(self):
        self.assertEqual(fuse([['a','a','b']]), fuse([['a','b']]))

    def test_consensus_and_determinism(self):
        ranks = [['a','b'], ['c','b']]
        self.assertEqual([h['id'] for h in fuse(ranks)], ['b','a','c'])
        self.assertEqual(fuse(ranks, 1), fuse(ranks)[:1])

    def test_empty(self):
        self.assertEqual(fuse([[],[]]), [])

    def test_group_penalty(self):
        ranks = [['a','b','c']]
        self.assertEqual([h['id'] for h in fuse(ranks, groups={'a':1,'b':1,'c':2}, diversity=.1)], ['a','c','b'])
