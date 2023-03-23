
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

/**
 * @brief A transportation solver for unidimensional optimal transport
 */
class Transportation1d {
 public:
  using Solution = std::vector<std::tuple<int, int, long long>>;
  using Event = std::pair<long long, long long>;
  using PrioQueue = std::priority_queue<Event>;
  /**
   * @brief Initialize the datastructure
   *
   * @param u Source positions
   * @param v Sink positions
   * @param s Source supplies
   * @param d Sink demands
   */
  Transportation1d(const std::vector<long long> &u,
                   const std::vector<long long> &v,
                   const std::vector<long long> &s,
                   const std::vector<long long> &d);

  /**
   * @brief Number of sources in the problem
   */
  int nbSources() const { return u.size(); }

  /**
   * @brief Number of sinks in the problem
   */
  int nbSinks() const { return v.size(); }

  /**
   * @brief Compute the total supply at the sources
   */
  long long totalSupply() const { return S.back(); }

  /**
   * @brief Compute the total demand at the sinks
   */
  long long totalDemand() const { return D.back(); };

  /**
   * @brief Compute the cost of allocating a source to a sink
   */
  long long cost(int i, int j) const { return std::abs(u[i] - v[j]); }

  /**
   * @brief Compute the change in reduced cost at a boundary
   */
  long long delta(int i, int j) const {
    return cost(i, j + 1) + cost(i + 1, j) - cost(i + 1, j + 1) - cost(i, j);
  }

  /**
   * @brief Run the whole optimization with efficient algorithm
   */
  Solution solve();

  /**
   * @brief Compute the cost of a solution
   */
  long long cost(const Solution &sol) const;

  /**
   * @brief Check that the given solution is valid
   */
  void checkSolutionValid(const Solution &sol) const;

  /**
   * @brief Check that the given solution is optimal
   */
  void checkSolutionOptimal(const Solution &sol) const;

  /**
   * @brief Check the datastructure
   */
  void check() const;

  /**
   * @brief Read a serialized problem
   */
  static Transportation1d read(std::istream &f);

  /**
   * @brief Read a serialized solution
   */
  static Solution readSolution(std::istream &f);

  /**
   * @brief Write a serialized problem
   */
  void write(std::ostream &f) const;

  /**
   * @brief Write a serialized solution
   */
  static void writeSolution(const Solution &sol, std::ostream &f);

 private:
  /**
   * @brief Initialize the additional datastructures
   */
  void setupData();

  /**
   * @brief Flush the positions according to the non-overlap constraints
   */
  void flushPositions();

  /**
   * @brief Compute the current solution
   */
  Solution computeSolution();

  /**
   * @brief Push a single source
   */
  void push(int i);

  /**
   * @brief Push a single source
   */
  void pushOnce(int i);

  /**
   * @brief Push to the last sink to be used
   */
  void pushToLastSink(int i);

  /**
   * @brief Push to the next sink to be used
   */
  void pushToNewSink(int i);

  /**
   * @brief Push the events corresponding to a given source
   */
  void pushNewSourceEvents(int i);

  /**
   * @brief Push the events corresponding to entering a new sink
   */
  void pushNewSinkEvents(int i, int j);

  /**
   * @brief Get the reduced cost at the current position
   */
  long long getSlope(bool pop = false);

  /**
   * @brief Update the optimal sink
   */
  void updateOptimalSink(int i);

 private:
  // Problem data
  std::vector<long long> u;
  std::vector<long long> v;
  std::vector<long long> s;
  std::vector<long long> d;

  // Additional data
  std::vector<long long> S;
  std::vector<long long> D;

  // Solution and state
  std::vector<long long> p;
  PrioQueue events;
  long long lastPosition;
  int lastOccupiedSink;
  int optimalSink;
};
