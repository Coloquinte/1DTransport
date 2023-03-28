import numpy as np
import heapq
import math
import bisect


class TransportationProblem:
    """
    Representation and basic solution for a 1D transportation problem
    """

    def __init__(self, u, v, s, d):
        self.source_pos = u
        self.sink_pos = v
        self.source_supply = s
        self.sink_demand = d
        self.nb_sources = len(self.source_pos)
        self.nb_sinks = len(self.sink_pos)
        self.check()

    def check(self):
        """
        Check that the problem meets all preconditions
        """
        for a in [
            self.source_pos,
            self.sink_pos,
            self.source_supply,
            self.sink_demand,
        ]:
            assert isinstance(a, list)
            for b in a:
                assert isinstance(b, int)
        for a in [self.source_pos, self.source_supply]:
            assert len(a) == self.nb_sources
        for a in [self.sink_pos, self.sink_demand]:
            assert len(a) == self.nb_sinks
        # Non-negative supply/demand
        assert all(d >= 0 for d in self.source_supply)
        assert all(d >= 0 for d in self.sink_demand)

    def check_solution(self, x):
        """
        Check a solution presented as a collection of (source, sink, alloc) tuples
        """
        used_supply = np.zeros(self.nb_sources, dtype=np.int64)
        used_demand = np.zeros(self.nb_sinks, dtype=np.int64)
        for i, j, a in x:
            assert a >= 0
            used_supply[i] += a
            used_demand[j] += a
        assert np.all(used_supply == self.source_supply)
        assert np.all(used_demand <= self.sink_demand)
    
    def cost(self, i, j):
        return abs(self.source_pos[i] - self.sink_pos[j])

    def solution_cost(self, x):
        c = 0
        for i, j, a in x:
            c += a * self.cost(i, j)
        return c

    @property
    def total_supply(self):
        return sum(self.source_supply)

    @property
    def total_demand(self):
        return sum(self.sink_demand)
    
    @staticmethod
    def make_random(
        n, m, supply_ratio=10.0, demand_ratio=15.0, coord_ratio=10.0, unique=True, sorted_coords=True, nonzero=True, seed=None
    ):
        """
        Create a random problem
        """
        if coord_ratio < 1.0:
            raise RuntimeError("Should have coords per element larger than 1")
        if supply_ratio < 1.0 or demand_ratio < 1.0:
            raise RuntimeError(
                "Should have supply and demand per element larger than 1"
            )
        if supply_ratio > demand_ratio:
            raise RuntimeError("Should have larger demand than supply")
        tot_supply = math.ceil(supply_ratio * (n + m))
        tot_demand = math.ceil(demand_ratio * (n + m))
        coord_range = math.ceil(coord_ratio * (n + m))
        assert coord_range >= n
        assert coord_range >= m
        assert tot_supply >= n
        assert tot_demand >= m
        assert tot_supply <= tot_demand
        rng = np.random.default_rng(seed)
        u = rng.choice(coord_range, n, replace=not unique).tolist()
        if sorted_coords:
            u.sort()
        v = rng.choice(coord_range, m, replace=not unique).tolist()
        if sorted_coords:
            v.sort()
        # Supply/demand election so it's non zero and sums right
        s = TransportationProblem._make_random_capa(rng, n, tot_supply, nonzero).tolist()
        d = TransportationProblem._make_random_capa(rng, m, tot_demand, nonzero).tolist()
        pb = TransportationProblem(u, v, s, d)
        pb.check()
        return pb

    def to_string(self):
        us = " ".join(str(i) for i in self.source_pos)
        vs = " ".join(str(i) for i in self.sink_pos)
        ss = " ".join(str(i) for i in self.source_supply)
        ds = " ".join(str(i) for i in self.sink_demand)
        return f"{self.nb_sources} {self.nb_sinks}\n{us}\n{vs}\n{ss}\n{ds}\n"

    @staticmethod
    def solution_to_string(sol):
        s = "\n".join(f"{i} {j} {a}" for i, j, a in sol)
        return f"{len(sol)}\n{s}"

    @staticmethod
    def _make_random_capa(rng, n, tot, nonzero):
        tot_capa = tot - n if nonzero else tot
        capa = list(np.sort(rng.choice(tot_capa, n - 1, replace=True)))
        capa.insert(0, 0)
        capa.append(tot_capa)
        capa = np.diff(capa)
        if nonzero:
            return capa + 1
        return capa

    def solve(self):
        return BaselineSolver.solve(
            self.source_pos, self.sink_pos, self.source_supply, self.sink_demand
        )

class BaselineSolver(TransportationProblem):
    @staticmethod
    def solve(u, v, s, d):
        return BaselineSolver(u, v, s, d).solve_impl()

    def solve_impl(self):
        """
        Solve using the generic but inefficient solver
        """
        import networkx as nx

        n = self.nb_sources
        m = self.nb_sinks
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i, demand=-self.source_supply[i])
        for j in range(m):
            G.add_node(n + j, demand=self.sink_demand[j])
        if self.total_supply < self.total_demand:
            G.add_node(-1, demand=self.total_supply - self.total_demand)
            for j in range(m):
                G.add_edge(-1, n + j, weight=0)
        for i in range(n):
            for j in range(m):
                G.add_edge(i, n + j, weight=self.cost(i, j))
        flow = nx.min_cost_flow(G)
        x = []
        for i in range(n):
            if i not in flow:
                continue
            for j in range(m):
                if n + j not in flow[i]:
                    continue
                a = flow[i][n + j]
                if a > 0:
                    x.append((i, j, a))
        self.check_solution(x)
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and solve a transportation problem",
    )
    parser.add_argument("--sources", help="Number of sources", type=int, default=20)
    parser.add_argument("--sinks", help="Number of sinks", type=int, default=20)
    parser.add_argument(
        "--avg-supply", help="Average supply per element", type=float, default=10.0
    )
    parser.add_argument(
        "--avg-demand", help="Average demand per element", type=float, default=15.0
    )
    parser.add_argument(
        "--avg-coords", help="Average coordinates per element", type=float, default=10.0
    )
    parser.add_argument(
        "--sorted-coords", help="Generate a problem with positions sorted", action="store_true"
    )
    parser.add_argument(
        "--unique-coords", help="Generate a problem with unique positions", action="store_true"
    )
    parser.add_argument(
        "--nonzero-capacities", help="Generate a problem with nonzero supply and demands", action="store_true"
    )
    parser.add_argument("--seed", help="Random seed", type=int)
    parser.add_argument(
        "--solve", help="Solve and write the solution", action="store_true"
    )

    args = parser.parse_args()

    pb = TransportationProblem.make_random(
        args.sources,
        args.sinks,
        supply_ratio=args.avg_supply,
        demand_ratio=args.avg_demand,
        coord_ratio=args.avg_coords,
        sorted_coords=args.sorted_coords,
        unique=args.unique_coords,
        nonzero=args.nonzero_capacities,
        seed=args.seed,
    )
    print(pb.to_string())
    if args.solve:
        sol = pb.solve()
        pb.check_solution(sol)
        print(TransportationProblem.solution_to_string(sol))
