from transportation import FastSolver
from transportation import TransportationProblem

import unittest


class TestTransportation(unittest.TestCase):
    def test_naive(self):
        for i in range(10):
            pb = TransportationProblem.make_random(20, 20, seed=i)
            x1 = pb.solve_baseline()
            x2 = pb.solve_naive()
            self.assertEqual(pb.dense_solution_cost(x1), pb.dense_solution_cost(x2))

    def test_fast(self):
        for i in range(10):
            pb = TransportationProblem.make_random(20, 20, seed=i)
            x1 = pb.solve_naive()
            x2 = pb.solve()
            self.assertEqual(pb.dense_solution_cost(x1), pb.sparse_solution_cost(x2))

    def test_nonzero_bound(self):
        for i in range(10):
            pb = TransportationProblem.make_random(20, 20, seed=i)
            pb.check_nonzero_delta_bound()

    def test_delta_range(self):
        for k in range(10):
            pb = TransportationProblem.make_random(20, 20, seed=k)
            for i in range(pb.nb_sources-1):
                b, e = pb.nonzero_delta_range(i)
                for j in range(pb.nb_sinks-1):
                    if j < b:
                        assert pb.delta(i, j) == 0
                    elif j >= e:
                        assert pb.delta(i, j) == 0
                    else:
                        assert pb.delta(i, j) != 0

if __name__ == "__main__":
    unittest.main()
