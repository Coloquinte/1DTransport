from transportation import TransportationProblem

import unittest


class TestTransportation(unittest.TestCase):
    def test_random(self):
        for i in range(10):
            pb = TransportationProblem.make_random(20, 20)
            x1 = pb.solve_baseline()
            x2 = pb.solve_naive()
            self.assertEqual(pb.dense_solution_cost(x1), pb.dense_solution_cost(x2))


if __name__ == "__main__":
    unittest.main()
