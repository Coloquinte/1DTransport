import numpy as np
import heapq
import bisect


class TransportationProblem:
    """
    Representation and solution method for a 1D transportation problem
    """

    def __init__(self, u, v, s, d):
        self.source_pos = u
        self.sink_pos = v
        self.source_supply = s
        self.sink_demand = d
        self.prev_supply = np.cumsum(np.insert(self.source_supply, 0, 0)).tolist()
        self.prev_demand = np.cumsum(np.insert(self.sink_demand, 0, 0)).tolist()
        self.nb_sources = len(self.source_pos)
        self.nb_sinks = len(self.sink_pos)
        self.total_supply = np.sum(self.source_supply)
        self.total_demand = np.sum(self.sink_demand)
        self.sink_mid = [(self.sink_pos[i] + self.sink_pos[i+1]) // 2 for i in range(self.nb_sinks - 1)]
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
            self.prev_supply,
            self.prev_demand,
        ]:
            assert isinstance(a, list)
            for b in a:
                assert isinstance(b, int)
        for a in [self.source_pos, self.source_supply]:
            assert len(a) == self.nb_sources
        for a in [self.sink_pos, self.sink_demand]:
            assert len(a) == self.nb_sinks
        assert len(self.prev_supply) == self.nb_sources + 1
        assert len(self.prev_demand) == self.nb_sinks + 1
        # Strictly sorted
        for i in range(len(self.source_pos) - 1):
            assert self.source_pos[i] < self.source_pos[i+1]
        for i in range(len(self.sink_pos) - 1):
            assert self.sink_pos[i] < self.sink_pos[i+1]
        # Positive supply/demand
        assert all(d > 0 for d in self.source_supply)
        assert all(d > 0 for d in self.sink_demand)

    def check_dense_solution(self, x):
        """
        Check a solution presented as a source x sink 2D array
        """
        assert x.shape == (self.nb_sources, self.nb_sinks)
        assert np.all(x >= 0)
        assert np.all(x.sum(axis=1) == self.source_supply)
        assert np.all(x.sum(axis=0) <= self.sink_demand)

    def check_sparse_solution(self, x):
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

    def dense_to_sparse(self, x):
        ret = []
        for i in range(self.nb_sources):
            for j in range(self.nb_sinks):
                if x[i, j] != 0:
                    ret.append((i, j, x[i, j]))
        return ret

    def sparse_to_dense(self, x):
        ret = np.zeros((self.nb_sources, self.nb_sinks), dtype=np.int64)
        for i, j, a in x:
            ret[i, j] += a
        return ret

    def dense_solution_cost(self, x):
        return (x * self.full_cost_array()).sum()

    def sparse_solution_cost(self, x):
        c = 0
        for i, j, a in x:
            c += a * self.cost(i, j)
        return c

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
        u = np.sort(rng.choice(coord_range, n, replace=False)).tolist()
        v = np.sort(rng.choice(coord_range, m, replace=False)).tolist()
        # Supply/demand election so it's non zero and sums right
        s = TransportationProblem._make_random_capa(rng, n, tot_supply).tolist()
        d = TransportationProblem._make_random_capa(rng, m, tot_demand).tolist()
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

    def solve(self):
        return FastSolver.solve(
            self.source_pos, self.sink_pos, self.source_supply, self.sink_demand
        )

    def optimal_sink(self, i):
        """
        Return the optimal sink to allocate a source, irrespective of available supply and demand
        """
        return bisect.bisect_left(self.sink_mid, self.source_pos[i])

    def cost(self, i, j):
        """
        Return the cost at a given position
        """
        assert 0 <= i < self.nb_sources
        assert 0 <= j < self.nb_sinks
        return abs(self.source_pos[i] - self.sink_pos[j])

    def beta(self, i, j):
        """
        Return the value of beta at a given position

        This is the positional encoding of the source i being allocated first at sink j,
        or alternatively source i-1 being allocated last at sink j-1.
        See paper for more details.
        """
        assert 0 <= i <= self.nb_sources
        assert 0 <= j <= self.nb_sinks
        return self.prev_demand[j] - self.prev_supply[i]

    def delta(self, i, j):
        """
        Return the value of delta at a given position

        delta_{ij} = c_{i j} - c_{i j+1} - c_{i+1 j} + c_{i+1 j+1}
        See paper for more details.
        """
        assert 0 <= i < self.nb_sources - 1
        assert 0 <= j < self.nb_sinks - 1
        return (
            self.cost(i, j + 1)
            + self.cost(i + 1, j)
            - self.cost(i + 1, j + 1)
            - self.cost(i, j)
        )

    def nonzero_delta_range(self, i):
        """
        Return the range b <= j < e for which delta_{ij} is non-zero
        """
        # u_{i} >= v_{j+1} before
        range_min = bisect.bisect_right(self.sink_pos, self.source_pos[i]) - 1
        range_min = max(range_min, 0)
        # u_{i+1} <= v_{j} after
        range_max = bisect.bisect_left(self.sink_pos, self.source_pos[i + 1])
        range_max = min(range_max, self.nb_sinks - 1)
        return (range_min, range_max)

    def full_cost_array(self):
        """
        Return a complete cost array
        """
        return np.abs(np.reshape(self.source_pos, (-1, 1)) - self.sink_pos)

    def full_delta_array(self):
        """
        Return a complete delta array
        """
        return np.diff(np.diff(self.full_cost_array()), axis=0)

    def cleanup_encoding(self, encoding):
        """
        Obtain a non-decreasing encoding
        """
        encoding = list(encoding)
        p = self.total_demand - self.prev_supply[len(encoding)]
        for i in reversed(range(len(encoding))):
            assert p >= 0
            p = min(p, encoding[i])
            encoding[i] = p
        return encoding

    def encoding_to_solution(self, encoding):
        """
        Obtain the solution from a positional encoding
        """
        encoding = self.cleanup_encoding(encoding)
        ret = []
        i = 0
        j = 0
        while i < len(encoding) and j < self.nb_sinks:
            bi = self.prev_supply[i] + encoding[i]
            ei = self.prev_supply[i + 1] + encoding[i]
            bj = self.prev_demand[j]
            ej = self.prev_demand[j + 1]
            b = max(bi, bj)
            e = min(ei, ej)
            if e - b > 0:
                ret.append((i, j, e - b))
            if ei < ej:
                i += 1
            else:
                j += 1
        return ret


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
        self.check_dense_solution(x)
        return x


class NaiveSolver(TransportationProblem):
    @staticmethod
    def solve(u, v, s, d):
        """
        Solve using the simple successive shortest path method
        """
        return NaiveSolver(u, v, s, d).solve_impl()

    def solve_impl(self):
        x = np.zeros((self.nb_sources, self.nb_sinks), dtype=np.int64)
        for i in range(self.nb_sources):
            self.push(i, x)
        self.check_dense_solution(x)
        return x

    def push(self, i, x):
        while self.remaining_supply(i, x) > 0:
            self.push_once(i, x)

    def push_once(self, i, x):
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
        elif self.marginal_cost(i, j - 1, x) < self.cost(i, j):
            self.push_to_sink(i, j - 1, x)
        else:
            self.push_to_sink(i, j, x)

    def push_to_sink(self, i, j, x):
        capa = self.path_capacity(i, j, x)

        while self.remaining_demand(j, x) == 0:
            x[i, j] += capa
            i = np.nonzero(x[:, j])[0][0]
            assert x[i, j] >= capa
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


class FastSolver(TransportationProblem):
    @staticmethod
    def solve(u, v, s, d):
        """
        Solve using the optimized successive shortest path method
        """
        return FastSolver(u, v, s, d).solve_impl()

    def __init__(self, u, v, s, d):
        self.pos_encoding = []
        self.last_occupied_sink = 0
        self.last_position = 0
        self.events = []
        super().__init__(u, v, s, d)

    def solve_impl(self):
        for i in range(self.nb_sources):
            self.push(i)
        x = self.encoding_to_solution(self.pos_encoding)
        self.check_sparse_solution(x)
        return x

    def push(self, i):
        o = self.optimal_sink(i)
        self.push_new_source_events(i)
        # Earliest: placed right after the others or at the beginning of the optimal sink
        self.last_position = max(self.last_position, self.beta(i, o))
        self.push_new_sink_events(i, o)
        while self.last_position > self.beta(i + 1, self.last_occupied_sink + 1):
            self.push_once(i)
        self.pos_encoding.append(self.last_position)

    def push_once(self, i):
        j = self.last_occupied_sink
        if j == self.nb_sinks - 1:
            self.push_to_last_sink(i)
        else:
            marginal_cost_right = self.cost(i, j + 1)
            marginal_cost_left = self.get_slope() + self.cost(i, j)
            if self.last_position == 0:
                self.push_to_new_sink(i)
            elif marginal_cost_left >= marginal_cost_right:
                self.push_to_new_sink(i)
            else:
                self.push_to_last_sink(i)

    def push_to_last_sink(self, i):
        j = self.last_occupied_sink
        # Minimum position obtained pushing this source
        min_pos = max(self.beta(i + 1, j + 1), 0)
        # Slope at the current position
        slope = self.get_slope(pop=True)
        if len(self.events) == 0:
            # No more event: advance all the way
            self.last_position = min_pos
        else:
            # Event in the queue: advance until then
            self.last_position = max(min_pos, -self.events[0][0])
        heapq.heappush(self.events, (-self.last_position, slope))

    def push_to_new_sink(self, i):
        self.push_new_sink_events(i, self.last_occupied_sink + 1)

    def get_slope(self, pop=False):
        """
        Obtain the total slope at the current position
        """
        slope = 0
        while len(self.events) != 0 and self.events[0][0] == -self.last_position:
            slope += self.events[0][1]
            heapq.heappop(self.events)
        if not pop and slope != 0:
            heapq.heappush(self.events, (-self.last_position, slope))
        return slope

    def check(self):
        super().check()
        assert all(p >= 0 for p in self.pos_encoding)
        for i, p in enumerate(self.pos_encoding):
            assert p <= self.total_demand - self.prev_supply[i + 1]

    def push_new_source_events(self, i):
        """
        Push the events when adding a new source
        """
        if i == 0:
            return
        b, e = self.nonzero_delta_range(i - 1)
        for j in range(b, min(e, self.last_occupied_sink)):
            pos = self.beta(i, j + 1)
            d = self.delta(i - 1, j)
            heapq.heappush(self.events, (-pos, d))

    def push_new_sink_events(self, i, j):
        """
        Push the additional events when pushing to a new sink
        """
        if j <= self.last_occupied_sink:
            return
        for l in range(self.last_occupied_sink, j):
            pos = self.beta(i, l + 1)
            d = self.cost(i, l) - self.cost(i, l + 1)
            pos = min(pos, self.last_position)
            heapq.heappush(self.events, (-pos, d))
        self.last_occupied_sink = j
