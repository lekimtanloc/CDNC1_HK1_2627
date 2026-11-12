import math
import unittest

from week3_retrieval_metrics import evaluate, score_query


class RetrievalMetricTests(unittest.TestCase):
    def test_perfect_and_missing(self):
        self.assertEqual(score_query(["a", "b"], {"a", "b"}, 2), {
            "recall": 1.0, "ndcg": 1.0, "reciprocal_rank": 1.0,
        })
        self.assertEqual(score_query(["x"], {"a"}, 5), {
            "recall": 0.0, "ndcg": 0.0, "reciprocal_rank": 0.0,
        })

    def test_rank_and_missing_query(self):
        shifted = score_query(["x", "a"], {"a"}, 2)
        self.assertEqual(shifted["recall"], 1.0)
        self.assertAlmostEqual(shifted["ndcg"], 1 / math.log2(3))
        self.assertEqual(shifted["reciprocal_rank"], 0.5)
        macro = evaluate({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, 5)
        self.assertEqual(macro, {"recall": .5, "ndcg": .5, "reciprocal_rank": .5})

    def test_invalid_rankings(self):
        with self.assertRaises(ValueError):
            score_query(["a", "a"], {"a"}, 5)
        with self.assertRaises(ValueError):
            evaluate({"q": ["q"]}, {"q": {"a"}}, 5)


if __name__ == "__main__":
    unittest.main()
