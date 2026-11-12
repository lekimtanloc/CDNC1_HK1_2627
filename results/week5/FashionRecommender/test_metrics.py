import unittest,math
from metrics import score
class TestMetrics(unittest.TestCase):
 def test_empty(self):self.assertTrue(all(v==0 for v in score(['a'],set()).values()))
 def test_miss(self):self.assertTrue(all(v==0 for v in score(['a'],{'b'}).values()))
 def test_first(self):self.assertEqual(score(['a'],{'a'})['MRR@10'],1);self.assertEqual(score(['a'],{'a'})['NDCG@5'],1)
 def test_multiple(self):
  s=score(['a','x','b'],{'a','b','c'});self.assertAlmostEqual(s['NDCG@5'],1.5/(1+1/math.log2(3)+.5));self.assertEqual(s['Recall@5'],2/3)
 def test_tenth(self):
  s=score(list(map(str,range(10))),{'9'});self.assertEqual(s['MRR@10'],.1);self.assertEqual(s['NDCG@5'],0);self.assertAlmostEqual(s['NDCG@10'],1/math.log2(11))
 def test_outside(self):
  s=score(list(map(str,range(11))),{'10'});self.assertEqual(s['MRR@10'],0);self.assertEqual(s['NDCG@10'],0)
if __name__=='__main__':unittest.main()
