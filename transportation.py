import numpy as np
import heapq


class TransportationProblem:
    """
    Representation and solution method for a 1D transportation problem
    """

    def __init__(self, u, v, s, d):
        self.source_pos = np.array(u, dtype=np.int64)
        self.sink_pos = np.array(v, dtype=np.int64)
        self.source_supply = np.array(s, dtype=np.int64)
        self.sink_demand = np.array(d, dtype=np.int64)
        self.prev_supply = np.cumsum(np.insert(self.source_supply, 0, 0))
        self.prev_demand = np.cumsum(np.insert(self.sink_demand, 0, 0))
        self.nb_sources = len(self.source_pos)
        self.nb_sinks = len(self.sink_pos)
        self.total_supply = np.sum(self.source_supply)
        self.total_demand = np.sum(self.sink_demand)
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

    def solve(self):
        return FastSolver.solve(
            self.source_pos, self.sink_pos, self.source_supply, self.sink_demand
        )

    def optimal_sink(self, i):
        """
        Return the optimal sink to allocate a source, irrespective of available supply and demand
        """
        pos = self.source_pos[i]
        return np.argmin(np.abs(self.sink_pos - pos))

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
        return abs(self.source_pos[i] - self.sink_pos[j])

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

    def check_nonzero_delta_bound(self):
        nonzeros = (self.full_delta_array() != 0).sum()
        if nonzeros > self.nb_sources + self.nb_sinks - 3:
            raise RuntimeError("Theoretical nonzero bound exceeded")

    def cleanup_encoding(self, encoding):
        """
        Obtain a non-decreasing encoding
        """
        encoding = list(encoding)
        p = self.total_demand - self.total_supply
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

    def cleanup_events(self, events):
        """
        Sort and uniquify a set of (position, slope) events
        """
        d = dict()
        for p, s in events:
            if p in d:
                d[p] += s
            else:
                d[p] = s
        return sorted(d.items())

    def print_problem(self):
        print(f"Problem with {self.nb_sources} sources, {self.nb_sinks} sinks")
        print(f"Source pos: {list(self.source_pos)}")
        print(f"Sink pos: {list(self.sink_pos)}")
        print(f"Source supply: {list(self.source_supply)}")
        print(f"Sink demand: {list(self.sink_demand)}")


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
        self.print_problem()
        print("Solving with naive method")
        x = np.zeros((self.nb_sources, self.nb_sinks), dtype=np.int64)
        for i in range(self.nb_sources):
            print(f"Pushing source {i}, optimal sink {self.optimal_sink(i)}")
            while self.remaining_supply(i, x) > 0:
                self.push(i, x)
            minpos = self.positional_encoding(i, x)
            maxpos = self.positional_encoding(i, x, True)
            print(f"Pushed {i} to positional encoding {minpos}-{maxpos}")
            enc = [self.positional_encoding(k, x, True) for k in range(i + 1)]
            #print(f"Positions: {enc}")
            print(f"Cost after source {i}: {self.dense_solution_cost(x)}")
        self.check_dense_solution(x)
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

        print(f"\tPushing {capa} flow from {i} to {j}")
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

    def positional_encoding(self, i, x, incl=False):
        j = np.nonzero(x[i])[0][0]
        pos = 0
        for l in range(j + incl):
            pos += self.remaining_demand(l, x)
        return pos


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
        self.prio_queue = []
        super().__init__(u, v, s, d)

    def nb_placed_sources(self):
        return len(self.pos_encoding)

    def solve_impl(self):
        self.print_problem()
        print("Solving with fast method")
        for i in range(self.nb_sources):
            o = self.optimal_sink(i)
            self.push_new_source_events(i)
            # Earliest: placed right after the others or at the beginning of the optimal sink
            self.last_position = max(self.last_position, self.beta(i, o))
            self.push_new_sink_events(i, o)
            print(
                f"Pushing source {i}, optimal sink {o}, initial pos {self.last_position}"
            )
            #print(f"Positions: {self.cleanup_encoding(self.pos_encoding)}")
            #print(f"Events: {self.get_events()}")
            # While the new source would stick out in the next sink
            while self.last_position > self.beta(i + 1, self.last_occupied_sink + 1):
                #print(
                #    f"\tCurrent position is {self.last_position} but current sink "
                #    f"{self.last_occupied_sink} ends at {self.beta(i + 1, self.last_occupied_sink + 1)}: pushing"
                #)
                # Decide whether to push left or right
                self.decide_push(i, self.last_occupied_sink)
            self.pos_encoding.append(self.last_position)
            self.check()
            x = self.encoding_to_solution(self.pos_encoding)
            print(f"Cost after source {i}: {self.sparse_solution_cost(x)}")
        x = self.encoding_to_solution(self.pos_encoding)
        self.check_sparse_solution(x)
        return x

    def decide_push(self, i, j):
        next_sink_pos = self.beta(i + 1, self.last_occupied_sink + 1)
        assert self.last_position > next_sink_pos
        if j == self.nb_sinks - 1:
            self.push_to_last_sink(i, j)
            #print(f"\tPushed on sink {j}; new position is {self.last_position}")
        else:
            marginal_cost_right = self.cost(i, j + 1) - self.cost(i, j)
            marginal_cost_left = self.get_slope()
            if next_sink_pos <= 0 or marginal_cost_left >= marginal_cost_right:
                #print(f"\tReaching new sink {j+1}")
                self.push_to_new_sink(i, j + 1)
            else:
                self.push_to_last_sink(i, j)
                #print(f"\tPushed on sink {j}; new position is {self.last_position}")

    def push_to_last_sink(self, i, j):
        assert j == self.last_occupied_sink
        assert i == self.nb_placed_sources()
        # Update the bounds and position
        min_pos = max(self.beta(i + 1, j + 1), 0)
        slope = 0
        pos = self.last_position
        while len(self.prio_queue) > 0 and self.prio_queue[0][0] == -self.last_position:
            slope += self.prio_queue[0][1]
            heapq.heappop(self.prio_queue)
        if len(self.prio_queue) == 0:
            self.last_position = min_pos
        else:
            self.last_position = max(-self.prio_queue[0][0], min_pos)
        heapq.heappush(self.prio_queue, (-self.last_position, slope))

    def push_to_new_sink(self, i, j):
        assert j == self.last_occupied_sink + 1
        assert i == self.nb_placed_sources()
        self.push_new_sink_events(i, j)

    def get_events(self):
        """
        Return all events currently in the queue
        """
        return self.cleanup_events([(-p, s) for p, s in self.prio_queue])

    def get_slope(self):
        slope = 0
        while len(self.prio_queue) > 0 and self.prio_queue[0][0] == -self.last_position:
            slope += self.prio_queue[0][1]
            heapq.heappop(self.prio_queue)
        if slope != 0:
            heapq.heappush(self.prio_queue, (-self.last_position, slope))
        return slope

    def check(self):
        super().check()
        assert all(p >= 0 for p in self.pos_encoding)
        for i, p in enumerate(self.pos_encoding):
            assert p <= self.total_demand - self.prev_supply[i + 1]
        for p, s in self.prio_queue:
            assert -p <= self.last_position

    def push_new_source_events(self, i):
        """
        Push the events when adding a new source
        """
        if i == 0:
            return
        for j in range(self.last_occupied_sink):
            pos = self.beta(i + 1, j + 1)
            d = self.delta(i - 1, j)
            if pos > 0 and d != 0 and pos <= self.last_position:
                print(f"\t\tNew source event {i} {j} at {pos} slope {d}")
            self.push_event(pos, d)

    def push_new_sink_events(self, i, j):
        """
        Push the additional events when pushing to a new sink
        """
        if j <= self.last_occupied_sink:
            return
        for l in range(self.last_occupied_sink, j):
            pos = self.beta(i, j)
            d = self.cost(i, l) - self.cost(i, l + 1)
            if pos > 0 and d != 0 and pos <= self.last_position:
                print(f"\t\tNew sink event {i} {j} at {pos} slope {d}")
            self.push_event(pos, d)
        self.last_occupied_sink = j

    def push_event(self, pos, slope):
        if pos <= 0:
            # Unused event anyway
            return
        if slope == 0:
            # There's nothing there
            return
        if pos > self.last_position:
            # The event is already passed; happens when pushing to a new sink
            return
        heapq.heappush(self.prio_queue, (-pos, slope))
