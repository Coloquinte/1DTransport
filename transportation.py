import numpy as np


class TransportationProblem:
    """
    Representation and solution method for a 1D transportation problem
    """

    def __init__(self, u, v, s, d):
        self.source_pos = u
        self.sink_pos = v
        self.source_supply = s
        self.sink_demand = d
        self.prev_supply = np.cumsum(np.insert(self.source_supply, 0, 0))
        self.prev_demand = np.cumsum(np.insert(self.sink_demand, 0, 0))
        self.nb_sources = len(self.source_pos)
        self.nb_sinks = len(self.sink_pos)
        self.total_supply = np.sum(self.source_supply)
        self.total_demand = np.sum(self.sink_demand)
        self.check()

    def check(self):
        for a in [
            self.source_pos,
            self.sink_pos,
            self.source_supply,
            self.sink_demand,
            self.prev_supply,
            self.prev_demand,
        ]:
            assert isinstance(a, np.ndarray)
            assert np.issubdtype(a.dtype, np.integer)
        for a in [self.source_pos, self.source_supply]:
            assert a.shape == (self.nb_sources,)
        for a in [self.sink_pos, self.sink_demand]:
            assert a.shape == (self.nb_sinks,)
        assert self.prev_supply.shape == (self.nb_sources + 1,)
        assert self.prev_demand.shape == (self.nb_sinks + 1,)
        # Strictly sorted
        assert np.all(np.diff(self.source_pos) > 0)
        assert np.all(np.diff(self.sink_pos) > 0)
        # Positive supply/demand
        assert np.all(self.source_supply > 0)
        assert np.all(self.sink_demand > 0)

    def check_solution(self, x):
        assert x.shape == (self.nb_sources, self.nb_sinks)
        assert np.all(x >= 0)
        assert np.all(x.sum(axis=1) == self.source_supply)
        assert np.all(x.sum(axis=0) <= self.sink_demand)

    def solution_cost(self, x):
        return (x * self.full_cost_array()).sum()

    @staticmethod
    def make_random(
        n, m, tot_supply=None, tot_demand=None, coord_range=None, seed=None
    ):
        """
        Create a random problem
        """
        if tot_supply is None:
            tot_supply = 10 * (n + m)
        if tot_demand is None:
            tot_demand = 15 * (n + m)
        if coord_range is None:
            coord_range = 10 * (n + m)
        assert coord_range >= n
        assert coord_range >= m
        assert tot_supply >= n
        assert tot_demand >= m
        assert tot_supply <= tot_demand
        rng = np.random.default_rng(seed)
        u = np.sort(rng.choice(coord_range, n, replace=False))
        v = np.sort(rng.choice(coord_range, m, replace=False))
        # Supply/demand election so it's non zero and sums right
        s = TransportationProblem._make_random_capa(rng, n, tot_supply)
        d = TransportationProblem._make_random_capa(rng, m, tot_demand)
        return TransportationProblem(u, v, s, d)

    @staticmethod
    def _make_random_capa(rng, n, tot):
        capa = list(np.sort(rng.choice(tot - n, n - 1, replace=False)))
        capa.insert(0, 0)
        capa.append(tot - n)
        capa = np.diff(capa)
        return capa + 1

    def solve_baseline(self):
        return BaselineSolver.solve(
            self.source_pos, self.sink_pos, self.source_supply, self.sink_demand
        )

    def solve_naive(self):
        return NaiveSolver.solve(
            self.source_pos, self.sink_pos, self.source_supply, self.sink_demand
        )

    def optimal_sink(self, i):
        pos = self.source_pos[i]
        return np.argmin(np.abs(self.sink_pos - pos))

    def cost(self, i, j):
        assert 0 <= i < self.nb_sources
        assert 0 <= j < self.nb_sinks
        return abs(self.source_pos[i] - self.sink_pos[j])

    def delta(self, i, j):
        assert 0 <= i < self.nb_sources - 1
        assert 0 <= j < self.nb_sinks - 1
        return abs(self.source_pos[i] - self.sink_pos[j])

    def full_cost_array(self):
        return np.abs(np.reshape(self.source_pos, (-1, 1)) - self.sink_pos)

    def full_delta_array(self):
        """
        Return the complete delta array
        """
        return np.diff(np.diff(self.full_cost_array()), axis=0)

    def _check_nonzero_delta_bound(self):
        nonzeros = (self.full_delta_array() != 0).sum()
        if nonzeros > self.nb_sources + self.nb_sinks - 3:
            raise RuntimeError("Theoretical nonzero bound exceeded")


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
        c = self.full_cost_array()
        for i in range(n):
            for j in range(m):
                G.add_edge(i, n + j, weight=c[i, j])
        flow = nx.min_cost_flow(G)
        x = np.zeros((n, m), dtype=np.int64)
        for i in range(n):
            if i not in flow:
                continue
            for j in range(m):
                if n + j not in flow[i]:
                    continue
                x[i, j] = flow[i][n + j]
        self.check_solution(x)
        return x


class NaiveSolver(TransportationProblem):
    @staticmethod
    def solve(u, v, s, d):
        return NaiveSolver(u, v, s, d).solve_impl()

    def solve_impl(self):
        """
        Solve using the simple successive shortest path method
        """
        x = np.zeros((self.nb_sources, self.nb_sinks), dtype=np.int64)
        for i in range(self.nb_sources):
            while self.remaining_supply(i, x) > 0:
                self.push(i, x)
        self.check_solution(x)
        return x

    def push(self, i, x):
        opt = self.optimal_sink(i)
        free_demand_after = [
            l for l in range(opt, self.nb_sinks) if self.remaining_demand(l, x) > 0
        ]
        if len(free_demand_after) == 0:
            self.push_to_sink(i, self.nb_sinks - 1, x)
            return
        j = free_demand_after[0]
        free_demand_before = [l for l in range(0, j) if self.remaining_demand(l, x) > 0]
        if len(free_demand_before) == 0:
            self.push_to_sink(i, j, x)
        elif self.marginal_cost(i, j - 1, x) < self.marginal_cost(i, j, x):
            self.push_to_sink(i, j - 1, x)
        else:
            self.push_to_sink(i, j, x)

    def push_to_sink(self, i, j, x):
        capa = self.path_capacity(i, j, x)
        print(f"Pushing {capa} from {i} to {j}")
        while self.remaining_demand(j, x) == 0:
            x[i, j] += capa
            i = np.nonzero(x[:, j])[0][0]
            assert x[i, j] >= capa
            print(f"\tPushing {i} from {j} to {j-1}")
            x[i, j] -= capa
            j -= 1
        x[i, j] += capa
        assert (x >= 0).all()

    def marginal_cost(self, i, j, x):
        c = self.cost(i, j)
        while self.remaining_demand(j, x) == 0:
            i = np.nonzero(x[:, j])[0][0]
            assert x[i, j] > 0
            assert (x[:i, j] == 0).all()
            c += self.cost(i, j - 1) - self.cost(i, j)
            j -= 1
        return c

    def path_capacity(self, i, j, x):
        capa = self.remaining_supply(i, x)
        while self.remaining_demand(j, x) == 0:
            i = np.nonzero(x[:, j])[0][0]
            assert (x[:i, j] == 0).all()
            capa = min(capa, x[i, j])
            j -= 1
        capa = min(capa, self.remaining_demand(j, x))
        assert capa > 0
        return capa

    def remaining_supply(self, i, x):
        return self.source_supply[i] - x[i].sum()

    def remaining_demand(self, j, x):
        return self.sink_demand[j] - x[:, j].sum()


pb = TransportationProblem.make_random(100, 20)
x1 = pb.solve_baseline()
x2 = pb.solve_naive()
print(pb.solution_cost(x1))
print(pb.solution_cost(x2))
assert pb.solution_cost(x1) == pb.solution_cost(x2)
