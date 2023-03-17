from transportation import FastSolver
from transportation import TransportationProblem

import unittest


class TestTransportation(unittest.TestCase):
    def not_done_test_naive(self):
        for i in range(100):
            pb = TransportationProblem.make_random(100, 100, seed=i)
            x1 = pb.solve_baseline()
            x2 = pb.solve_naive()
            self.assertEqual(pb.dense_solution_cost(x1), pb.dense_solution_cost(x2))

    def test_fast(self):
        for i in range(100):
            pb = TransportationProblem.make_random(100, 100, seed=i)
            x1 = pb.solve_naive()
            x2 = pb.solve()
            self.assertEqual(pb.dense_solution_cost(x1), pb.sparse_solution_cost(x2))

    def test_nonzero_bound(self):
        for i in range(100):
            pb = TransportationProblem.make_random(20, 20, seed=i)
            pb.check_nonzero_delta_bound()


if __name__ == "__main__":
    unittest.main()
