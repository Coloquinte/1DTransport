
#include "transportation_1d.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

Transportation1d::Transportation1d(const std::vector<long long> &u,
                                   const std::vector<long long> &v,
                                   const std::vector<long long> &s,
                                   const std::vector<long long> &d)
    : u(u), v(v), s(s), d(d) {
  setupData();
  check();
}

void Transportation1d::setupData() {
  D.reserve(nbSinks() + 1);
  D.push_back(0LL);
  for (long long c : d) {
    D.push_back(D.back() + c);
  }
  S.reserve(nbSources() + 1);
  S.push_back(0LL);
  for (long long c : s) {
    S.push_back(S.back() + c);
  }
}

void Transportation1d::flushPositions() {
  // Flush constraints from the right
  long long maxPos = totalDemand() - totalSupply();
  for (int i = p.size() - 1; i >= 0; --i) {
    maxPos = std::min(p[i], maxPos);
    p[i] = maxPos;
  }
}

int Transportation1d::optimalSink(int i) const {
  int j = 0;
  while (j + 1 < nbSinks() && cost(i, j) >= cost(i, j + 1)) {
    ++j;
  }
  return j;
}

Transportation1d::Solution Transportation1d::solve() {
  
  return computeSolution();
}

Transportation1d::Solution
Transportation1d::computeSolution() {
  flushPositions();
  std::vector<std::tuple<int, int, long long>> ret;
  int i = 0;
  int j = 0;
  while (i < p.size() && j < nbSinks()) {
    long long bi = S[i] + p[i];
    long long ei = S[i + 1] + p[i];
    long long bj = D[j];
    long long ej = D[j + 1];
    long long b = std::max(bi, bj);
    long long e = std::min(ei, ej);
    if (e - b > 0) {
      ret.emplace_back(i, j, e - b);
    }
    if (ei < ej) {
      ++i;
    } else {
      ++j;
    }
  }
  return ret;
}

void Transportation1d::checkPositionsValid() const {
  if (p.empty()) return;
  if (p.front() < 0LL) {
    throw std::runtime_error("First source is scheduled too early");
  }
  int lastInd = p.size() - 1;
  if (p[lastInd] > totalDemand() - S[lastInd + 1]) {
    throw std::runtime_error("Last source is scheduled too late");
  }
  for (int i = 0; i + 1 < p.size(); ++i) {
    if (p[i] > p[i + 1]) {
      throw std::runtime_error("Positions overlap");
    }
  }
}

void Transportation1d::checkSolutionValid(const Solution &alloc) const {
  // Compute capacity usage
  std::vector<long long> usedSupply(nbSources());
  std::vector<long long> usedDemand(nbSinks());
  for (auto [i, j, a] : alloc) {
    usedSupply[i] += a;
    usedDemand[j] += a;
    if (a <= 0LL) {
      throw std::runtime_error("Allocation should be positive");
    }
  }
  for (int i = 0; i < nbSources(); ++i) {
    if (usedSupply[i] != s[i]) {
      throw std::runtime_error("Supply is not met");
    }
  }
  for (int j = 0; j < nbSinks(); ++j) {
    if (usedDemand[j] > d[j]) {
      throw std::runtime_error("Demand is not met");
    }
  }
}

void Transportation1d::checkSolutionOptimal(const Solution &alloc) const {
  // Compute capacity usage
  std::vector<long long> usedCap(nbSinks());
  for (auto [i, j, a] : alloc) {
    usedCap[j] += a;
  }

  // Compute the gain of moving sources to the right
  std::vector<long long> gainRight(nbSinks(),
                                   std::numeric_limits<long long>::min());
  for (auto [i, j, a] : alloc) {
    if (j + 1 < nbSinks()) {
      long long gain = cost(i, j) - cost(i, j + 1);
      gainRight[j] = std::max(gainRight[j], gain);
    }
  }

  // Compute the gain of moving sources to the left
  std::vector<long long> gainLeft(nbSinks(),
                                  std::numeric_limits<long long>::min());
  for (auto [i, j, a] : alloc) {
    if (j - 1 >= 0) {
      long long gain = cost(i, j) - cost(i, j - 1);
      gainLeft[j] = std::max(gainLeft[j], gain);
    }
  }

  // Does it allow some positive gain move right?
  for (int snk = 0; snk + 1 < nbSinks(); ++snk) {
    if (usedCap[snk] == 0LL) {
      continue;
    }
    long long gain = gainRight[snk];
    for (int nxt = snk + 1; nxt < nbSinks(); ++nxt) {
      if (usedCap[nxt] < d[nxt]) {
        if (gain > 0) {
          throw std::runtime_error("Found an improving right move");
        }
        snk = nxt - 1;
        break;
      }
      gain += gainRight[nxt];
    }
  }

  // Does it allow some positive gain move left?
  for (int snk = nbSinks() - 1; snk >= 1; --snk) {
    if (usedCap[snk] == 0LL) {
      continue;
    }
    long long gain = gainLeft[snk];
    for (int nxt = snk - 1; nxt >= 0; --nxt) {
      if (usedCap[nxt] < d[nxt]) {
        if (gain > 0) {
          throw std::runtime_error("Found an improving left move");
        }
        snk = nxt + 1;
        break;
      }
      gain += gainLeft[nxt];
    }
  }
}

void Transportation1d::check() const {
  if (u.size() != nbSources()) {
    throw std::runtime_error("Inconsistant source positions");
  }
  if (v.size() != nbSinks()) {
    throw std::runtime_error("Inconsistant sink positions");
  }
  if (s.size() != nbSources()) {
    throw std::runtime_error("Inconsistant supplies");
  }
  if (d.size() != nbSinks()) {
    throw std::runtime_error("Inconsistant demands");
  }
  if (S.size() != nbSources() + 1) {
    throw std::runtime_error("Inconsistant total supplies");
  }
  if (D.size() != nbSinks() + 1) {
    throw std::runtime_error("Inconsistant total demands");
  }
  for (long long c : s) {
    if (c <= 0) {
      throw std::runtime_error("Supplies must be positive");
    }
  }
  for (long long c : d) {
    if (c <= 0) {
      throw std::runtime_error("Demands must be positive");
    }
  }
  if (totalSupply() > totalDemand()) {
    throw std::runtime_error("The supply should be no larger than the demand");
  }
  if (!std::is_sorted(u.begin(), u.end())) {
    throw std::runtime_error("Source positions should be sorted");
  }
  if (!std::is_sorted(v.begin(), v.end())) {
    throw std::runtime_error("Sink positions should be sorted");
  }
  if (p.size() > nbSources()) {
    throw std::runtime_error("Too many positions computed");
  }
}

Transportation1d Transportation1d::read(std::istream &f) {
  int nbSources;
  int nbSinks;
  f >> nbSources >> nbSinks;
  long long x;
  std::vector<long long> u, v, s, d;
  for (int i = 0; i < nbSources; ++i) {
    f >> x;
    u.push_back(x);
  }
  for (int i = 0; i < nbSinks; ++i) {
    f >> x;
    v.push_back(x);
  }
  for (int i = 0; i < nbSources; ++i) {
    f >> x;
    s.push_back(x);
  }
  for (int i = 0; i < nbSinks; ++i) {
    f >> x;
    d.push_back(x);
  }
  return Transportation1d(u, v, s, d);
}

void Transportation1d::write(std::ostream &f) const {
  f << nbSources() << " " << nbSinks() << std::endl;
  for (int i = 0; i < nbSources(); ++i) {
    if (i > 0) f << " ";
    f << u[i];
  }
  f << std::endl;
  for (int i = 0; i < nbSinks(); ++i) {
    if (i > 0) f << " ";
    f << v[i];
  }
  f << std::endl;
  for (int i = 0; i < nbSources(); ++i) {
    if (i > 0) f << " ";
    f << s[i];
  }
  f << std::endl;
  for (int i = 0; i < nbSinks(); ++i) {
    if (i > 0) f << " ";
    f << d[i];
  }
  f << std::endl;
}

void Transportation1d::writeSolution(const Solution &sol, std::ostream &f) {
  f << sol.size() << std::endl;
  for (auto [i, j, a] : sol) {
    f << i << " " << j << " " << a << std::endl;
  }
};